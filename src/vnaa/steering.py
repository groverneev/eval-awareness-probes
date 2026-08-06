"""Activation-addition steering along a frozen direction.

The intervention is deliberately small and mechanical so it can be tested
without a GPU: install one forward hook that adds ``coefficient * direction``
to the residual stream at a single layer, and remove it afterwards.

Two decisions are baked in here because each is a place the steering arm could
go quietly wrong:

- **Layer indexing matches the probe.** ``output_hidden_states`` returns
  ``n_layers + 1`` tensors where index 0 is the embedding output and index ``k``
  is the output of block ``k - 1``.  Every probe result in this project is
  reported against that index, so steering "at layer L" must hook block
  ``L - 1``.  Hooking ``layers[L]`` instead would inject one block later than
  the layer the direction was fitted at, and nothing downstream would complain.

- **Coefficient 0 installs no hook.** The unsteered condition is then
  bit-identical to a plain run rather than merely numerically close, so any
  difference at alpha = 0 is a bug rather than a rounding story.

During cached generation the hook fires once per forward pass.  A newly
computed token is therefore steered exactly once, and tokens already in the KV
cache keep the offset they were computed with, which is the intended semantics:
the model generates in a persistently shifted state.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def resolve_block(model, layer: int):
    """The module whose output is ``hidden_states[layer]``.

    Layer 0 is the embedding output and is rejected.  Adding a constant vector
    there is a disguised prompt edit, not an intervention on a representation
    the model computed, so it cannot support a causal claim.
    """
    blocks = getattr(getattr(model, "model", None), "layers", None)
    if blocks is None:
        raise AttributeError("model has no .model.layers to hook")
    if layer == 0:
        raise ValueError(
            "refusing to steer the embedding layer: that is a prompt edit in "
            "disguise, not an intervention on a computed representation"
        )
    if not 1 <= layer <= len(blocks):
        raise ValueError(f"layer {layer} out of range 1..{len(blocks)}")
    return blocks[layer - 1]


@contextmanager
def steer(model, layer: int, direction, coefficient: float):
    """Temporarily add ``coefficient * direction`` at ``layer``'s output.

    ``direction`` is used exactly as given; scaling into interpretable units is
    the caller's job (see ``run_steering``), so that what gets added is always
    visible at the call site rather than hidden behind a normalization here.
    """
    if coefficient == 0.0:
        yield model
        return

    block = resolve_block(model, layer)
    vector = torch.as_tensor(direction, dtype=torch.float32).flatten()
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is not None and vector.numel() != hidden_size:
        raise ValueError(
            f"direction has {vector.numel()} dims but the model's hidden size "
            f"is {hidden_size}"
        )

    def hook(_module, _inputs, output):
        # transformers 5.x decoder layers return a bare tensor; 4.x returned a
        # tuple.  Support both so a version bump cannot silently no-op this.
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        delta = (coefficient * vector).to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden + delta
        return (steered,) + tuple(output[1:]) if is_tuple else steered

    handle = block.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()
