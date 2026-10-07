# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pure-Python EDR gradient-anchor bucketing shared by model code."""

from __future__ import annotations

EDR_GRADIENT_ANCHOR_BUCKETS = (64, 128, 256, 512)


def edr_gradient_anchor_slots(selected_count: int, num_anchors: int) -> int:
    """Return the smallest bounded query bucket that holds an EDR sample."""
    if selected_count < 1 or selected_count > num_anchors:
        raise ValueError(
            "selected EDR anchors must be in [1, num_anchors], "
            f"got {selected_count} with num_anchors={num_anchors}"
        )
    for bucket in (*EDR_GRADIENT_ANCHOR_BUCKETS, num_anchors):
        slots = min(bucket, num_anchors)
        if selected_count <= slots:
            return slots
    raise RuntimeError("failed to choose an EDR gradient anchor bucket")


def edr_combined_gradient_anchor_slots(
    selected_count: int,
    num_anchors: int,
    max_slots: int,
) -> int:
    """Pad one combined gradient tail without padding its source horizons.

    Same-row horizons are compacted before batching.  The final partial batch
    keeps the 64-anchor shape granularity (or ``num_anchors`` when it
    is smaller), while full batches need no padding.  ``max_slots`` is the
    bounded combined-batch capacity.
    """
    if num_anchors < 1:
        raise ValueError(f"num_anchors must be >= 1, got {num_anchors}")
    if max_slots < num_anchors:
        raise ValueError(
            "combined EDR gradient capacity must be >= num_anchors, "
            f"got {max_slots} < {num_anchors}"
        )
    if selected_count < 1 or selected_count > max_slots:
        raise ValueError(
            "combined selected EDR anchors must be in [1, max_slots], "
            f"got {selected_count} with max_slots={max_slots}"
        )

    alignment = min(EDR_GRADIENT_ANCHOR_BUCKETS[0], num_anchors)
    rounded = ((selected_count + alignment - 1) // alignment) * alignment
    return min(rounded, max_slots)
