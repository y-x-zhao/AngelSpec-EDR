"""Tests for the dfly (DFlareV2) draft model.

dfly is a DFlash-family drafter (its own ``DFlyConfig`` / ``"Qwen3DFlyModel"`` →
``DFlyTrainer`` / ``DFlyModel``) so the hidden-states correction runs through the
wrapper hook. It is a ``DFlareDraftModel`` whose layers are restored to DFlash
shared-KV layers, with the DFlash FC ``context_proj`` re-added and the DFlare
per-layer fusion applied as a residual. It has no dependency on DSpark.

Covers:
1. auto/dispatch: ``DFlyConfig`` → ``DFlyDraftModel`` (and a plain ``DSparkConfig``
   still → ``DSparkDraftModel`` — no cross-routing).
2. Structure: DFlash shared-KV layers (no separate ``k_proj_target``), the
   re-added ``context_proj``, the inherited ``layer_fusion_weights`` / ``context_norm``.
3. hidden_correction: present, zero-init identity, and NO markov / confidence head.
4. ``target_hidden_size != hidden_size`` raises.
5. Tiny forward through the ``DFlyModel`` wrapper: 6-tuple, finite loss, all
   proposal slots learned, ``loss_components`` keys; and the correction actually
   runs (perturbing ``down_proj`` changes the loss).
6. state_dict carries the expected backbone/correction keys and none for the
   (absent) markov / confidence heads.
"""

import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from angelspec.models.dfly import DFlyModel
from angelspec.models.draft.auto import AutoEagle3DraftModel
from angelspec.models.draft.dflare import DFlareDraftModel
from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel

H, V, BS = 64, 128, 4


def _base(**kw):
    base = dict(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=V,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_target_layers=2,
        target_hidden_size=H,
        target_num_hidden_layers=12,
        mask_token_id=V - 1,
        # dfly carries no markov / confidence head (matches the shipped config).
        markov_rank=0,
        enable_confidence_head=False,
        confidence_head_with_markov=False,
        block_size=BS,
    )
    base.update(kw)
    return base


def _config(**kw):
    kw.setdefault("enable_hidden_correction", True)
    return DFlyConfig(**_base(**kw))


class TestDflyDispatchAndStructure(unittest.TestCase):
    def test_auto_dispatch(self):
        fly = AutoEagle3DraftModel.from_config(_config())
        self.assertIsInstance(fly, DFlyDraftModel)
        # A plain DSpark config still builds the DSpark drafter (no cross-routing).
        ds = AutoEagle3DraftModel.from_config(DSparkConfig(**_base()))
        self.assertIsInstance(ds, DSparkDraftModel)
        self.assertNotIsInstance(ds, DFlyDraftModel)

    def test_shared_kv_layers(self):
        # dfly restores DFlash shared-KV layers: k_proj/v_proj present, and NOT
        # DFlare's separate context projections (k_proj_target / v_proj_target).
        fly = DFlyDraftModel(_config())
        self.assertEqual(len(fly.layers), fly.num_layers)
        for layer in fly.layers:
            self.assertTrue(hasattr(layer.self_attn, "k_proj"))
            self.assertTrue(hasattr(layer.self_attn, "v_proj"))
            self.assertFalse(hasattr(layer.self_attn, "k_proj_target"))
            self.assertFalse(hasattr(layer.self_attn, "v_proj_target"))

    def test_context_proj_and_fusion_wired(self):
        cfg = _config()
        fly = DFlyDraftModel(cfg)
        # FC context projection re-added (DFlare.__init__ had deleted it).
        self.assertTrue(hasattr(fly, "context_proj"))
        self.assertEqual(
            tuple(fly.context_proj.weight.shape),
            (H, cfg.num_target_layers * H),
        )
        # DFlare fusion + norm inherited.
        self.assertEqual(
            tuple(fly.layer_fusion_weights.shape),
            (cfg.num_hidden_layers, cfg.num_target_layers),
        )
        self.assertTrue(hasattr(fly, "context_norm"))

    def test_context_cache_capability_matches_attention_layout(self):
        self.assertFalse(DFlareDraftModel.supports_context_cache)
        self.assertTrue(DFlyDraftModel.supports_context_cache)

    def test_hidden_correction_present_and_identity(self):
        fly = DFlyDraftModel(_config())
        self.assertIsNotNone(fly.hidden_correction)
        self.assertTrue(
            (fly.hidden_correction.down_proj.weight == 0).all(),
            "down_proj must be zero-init (identity at init)",
        )
        # dfly builds no markov / confidence head.
        self.assertIsNone(getattr(fly, "markov_head", None))
        self.assertIsNone(getattr(fly, "confidence_head", None))

    def test_hidden_correction_none_when_disabled(self):
        fly = DFlyDraftModel(_config(enable_hidden_correction=False))
        self.assertIsNone(fly.hidden_correction)
        self.assertFalse(any("hidden_correction" in k for k in fly.state_dict()))

    def test_target_hidden_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            DFlyDraftModel(_config(target_hidden_size=H + 8))


class TestDflyForward(unittest.TestCase):
    def _build_wrapper(self):
        torch.manual_seed(0)
        fly = AutoEagle3DraftModel.from_config(_config()).to(torch.float32)
        fly.freeze_embedding()
        m = DFlyModel(
            draft_model=fly,
            block_size=BS,
            num_anchors=6,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
        )
        m.eval()
        return m

    def _inputs(self):
        B, S = 2, 24
        g = torch.Generator().manual_seed(1)
        return dict(
            input_ids=torch.randint(0, V, (B, S), generator=g),
            hidden_states_list=[torch.randn(B, S, H, generator=g) for _ in range(2)],
            loss_mask=torch.ones(B, S),
            lm_head_weight=torch.randn(V, H, generator=g),
            last_hidden_states=torch.randn(B, S, H, generator=g),
        )

    def test_forward_six_tuple(self):
        m = self._build_wrapper()
        out = m(**self._inputs())
        self.assertEqual(len(out), 6)
        loss, _, lpp, _, cpp, comps = out
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(lpp.shape[0], BS)
        self.assertTrue((cpp > 0).all())
        # DFly rides the plain DFlash loss (no confidence head).
        self.assertEqual(
            set(comps),
            {"ce_loss", "kl_loss", "lk_loss", "l1_loss", "e2e_tv_loss"},
        )

    def test_correction_actually_runs(self):
        # With zero-init down_proj the correction is identity; perturbing it must
        # change the loss, proving the wrapper hook applies hidden_correction.
        m = self._build_wrapper()
        inputs = self._inputs()
        base_loss = m(**inputs)[0].item()
        with torch.no_grad():
            m.draft_model.hidden_correction.down_proj.weight.normal_()
        perturbed_loss = m(**inputs)[0].item()
        self.assertNotAlmostEqual(base_loss, perturbed_loss, places=5)

    def test_context_kv_cache_matches_forward_and_is_reused(self):
        torch.manual_seed(5)
        fly = DFlyDraftModel(_config()).to(torch.float32).eval()
        batch_size, context_length, draft_length = 1, 12, 8
        hidden_states = [
            torch.randn(batch_size, context_length, H) for _ in range(fly.num_target_layers)
        ]
        context_feature = fly.extract_context_feature(hidden_states)
        context_positions = torch.arange(context_length).unsqueeze(0)
        draft_positions = torch.arange(4, 4 + draft_length).unsqueeze(0)
        noise_embedding = torch.randn(batch_size, draft_length, H)

        context_projection_calls = 0

        def count_context_projection(_module, _inputs, _output):
            nonlocal context_projection_calls
            context_projection_calls += 1

        hook = fly.context_proj.register_forward_hook(count_context_projection)
        try:
            with torch.no_grad():
                expected = fly(
                    draft_input_ids=None,
                    context_feature=context_feature,
                    draft_position_ids=draft_positions,
                    context_position_ids=context_positions,
                    block_mask=None,
                    noise_embedding=noise_embedding,
                )
                context_cache = fly.prepare_context_cache(
                    context_feature,
                    context_positions,
                )
                actual = fly.forward_with_context_cache(
                    draft_input_ids=None,
                    context_cache=context_cache,
                    draft_position_ids=draft_positions,
                    context_position_ids=context_positions,
                    block_mask=None,
                    noise_embedding=noise_embedding,
                )
                repeated = fly.forward_with_context_cache(
                    draft_input_ids=None,
                    context_cache=context_cache,
                    draft_position_ids=draft_positions,
                    context_position_ids=context_positions,
                    block_mask=None,
                    noise_embedding=noise_embedding,
                )
        finally:
            hook.remove()

        self.assertEqual(context_projection_calls, 2)
        self.assertEqual(len(context_cache.layer_caches), fly.num_layers)
        self.assertTrue(all(not cache.key.requires_grad for cache in context_cache.layer_caches))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(repeated, expected, atol=2e-5, rtol=2e-5)

    def test_context_cache_segments_preserve_absolute_rope_positions(self):
        torch.manual_seed(6)
        fly = DFlyDraftModel(_config()).to(torch.float32).eval()
        context_length, split = 9, 4
        hidden_states = [torch.randn(1, context_length, H) for _ in range(fly.num_target_layers)]

        with torch.no_grad():
            full_feature = fly.extract_context_feature(hidden_states)
            full_cache = fly.prepare_context_cache(
                full_feature,
                torch.arange(context_length).unsqueeze(0),
            )
            prefix_feature = fly.extract_context_feature(
                [state[:, :split] for state in hidden_states]
            )
            prefix_cache = fly.prepare_context_cache(
                prefix_feature,
                torch.arange(split).unsqueeze(0),
            )
            suffix_feature = fly.extract_context_feature(
                [state[:, split:] for state in hidden_states]
            )
            suffix_cache = fly.prepare_context_cache(
                suffix_feature,
                torch.arange(split, context_length).unsqueeze(0),
                rope_sequence_length=context_length,
            )

        for full, prefix, suffix in zip(
            full_cache.layer_caches,
            prefix_cache.layer_caches,
            suffix_cache.layer_caches,
        ):
            torch.testing.assert_close(
                torch.cat((prefix.key, suffix.key), dim=2),
                full.key,
            )
            torch.testing.assert_close(
                torch.cat((prefix.value, suffix.value), dim=2),
                full.value,
            )

    def test_edr_reuses_batched_caches_across_rows_and_horizons(self):
        torch.manual_seed(7)
        fly = DFlyDraftModel(_config()).to(torch.float32)
        model = DFlyModel(
            draft_model=fly,
            block_size=BS - 1,  # BS physical query slots including the input anchor
            num_anchors=2,
            loss_objective="edr",
            edr_chunk_size=2,
            edr_vocab_chunk_size=17,
            query_includes_input_anchor=True,
        )
        uncached_model = deepcopy(model)
        uncached_model.draft_model.supports_context_cache = False
        sequence_length = 12
        batch = dict(
            input_ids=torch.randint(0, V, (2, sequence_length)),
            hidden_states_list=[
                torch.randn(2, sequence_length, H) for _ in range(fly.num_target_layers)
            ],
            loss_mask=torch.tensor(
                [
                    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
            attention_mask=torch.ones(2, sequence_length),
            lm_head_weight=torch.randn(V, H),
            last_hidden_states=torch.randn(2, sequence_length, H),
            target_norm=nn.Identity(),
        )
        cache_grad_modes = []
        prepare_context_cache = fly.prepare_context_cache

        def capture_cache(*args, **kwargs):
            cache_grad_modes.append(torch.is_grad_enabled())
            return prepare_context_cache(*args, **kwargs)

        fly.prepare_context_cache = capture_cache
        torch.manual_seed(29)
        optimized = model(**batch)
        torch.manual_seed(29)
        uncached = uncached_model(**batch)
        optimized[0].backward()
        uncached[0].backward()

        self.assertEqual(cache_grad_modes, [False, True])
        for actual, expected in zip(optimized[:5], uncached[:5]):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        optimized_grads = dict(fly.named_parameters())
        uncached_grads = dict(uncached_model.draft_model.named_parameters())
        self.assertEqual(optimized_grads.keys(), uncached_grads.keys())
        for name, parameter in optimized_grads.items():
            with self.subTest(parameter=name):
                expected_grad = uncached_grads[name].grad
                if expected_grad is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(
                        parameter.grad,
                        expected_grad,
                        atol=3e-5,
                        rtol=3e-5,
                    )
        self.assertIsNotNone(fly.context_proj.weight.grad)
        self.assertGreater(fly.context_proj.weight.grad.abs().sum().item(), 0.0)


class TestDflyCheckpointParamNames(unittest.TestCase):
    def test_state_dict_keys(self):
        fly = DFlyDraftModel(_config())
        keys = set(fly.state_dict())
        for expected in (
            "context_proj.weight",
            "layer_fusion_weights",
            "context_norm.weight",
            "hidden_correction.gate_proj.weight",
            "hidden_correction.up_proj.weight",
            "hidden_correction.down_proj.weight",
            "hidden_correction.hidden_norm.weight",
            "hidden_correction.embed_norm.weight",
        ):
            self.assertIn(expected, keys, f"missing key: {expected}")
        # No markov / confidence head params (dfly builds neither).
        self.assertFalse(any("markov_head" in k for k in keys))
        self.assertFalse(any("confidence_head" in k for k in keys))


if __name__ == "__main__":
    unittest.main()
