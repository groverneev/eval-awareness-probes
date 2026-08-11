"""Residual-stream activation extraction.

Slice 1 read path. Given conversations (lists of chat messages), run each
through an open-weight model with output_hidden_states=True and cache the
residual stream at every layer at one token position.

Decisions baked in here, each a place slice 1 could go quietly wrong:

- BOS: apply_chat_template already prepends exactly one BOS. We feed its ids
  straight to the model and never re-tokenize, so there is never a double BOS.

- Layer axis: hidden_states has n_layers + 1 entries. Index 0 is the embedding
  output (before any transformer block); indices 1..n_layers are block outputs.
  We keep all of them and record that index 0 is embeddings. Separation that
  is already present at layer 0 is lexical/format signal in the input, not
  something the model computes, which is exactly what the format-sensitivity
  question cares about.

- Read point: "last_user_token" templates with add_generation_prompt=False and
  reads the final token, which is the <|eot_id|> closing the last user turn.
  This is the literal "last token of the final user turn." The read point is a
  parameter because the multi-turn arm adds more of them.

- Batch size 1: we process one conversation at a time so the read index is
  always the true last token with no padding to reason about. Slice 1 is a few
  hundred short forward passes with no generation, so this is fast enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

Conversation = list[dict[str, str]]  # [{"role": "user"|"assistant"|"system", "content": ...}]

# Single-vector read points. A multi-turn conversation has no one obvious place
# that represents "what the model makes of this situation", so slice 3 probes
# several and reports each:
#   last_user_token       final token of the last user turn (the slice-1 point)
#   post_instruction_token end of the setup: last token of the system message if
#                         there is one, else of the first user turn. Tests
#                         whether the situation is recognized before any of the
#                         conversation proper has been read.
#   final_turn_mean       mean-pooled over the tokens of the final user turn,
#                         which is less hostage to one delimiter token.
# The per-turn trajectory is not here because it returns one vector per turn
# rather than one per conversation; see extract_trajectories.
READ_POINTS = ("last_user_token", "post_instruction_token", "final_turn_mean")

# Reconstruction compares conversations that end in an assistant reply.  Keep
# these separate from READ_POINTS: slice 3 deliberately ends on a user turn,
# and adding them there would make its existing runner silently read the
# preceding assistant reply instead.
ASSISTANT_READ_POINTS = ("final_assistant_token", "final_assistant_mean")


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LoadedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str
    n_layers: int  # transformer blocks, not counting the embedding layer
    hidden_size: int


def load_model(
    model_name: str,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    device_map: str | None = None,
    revision: str | None = None,
) -> LoadedModel:
    """Load tokenizer and model for activation extraction.

    dtype defaults to float32 for faithful activations on CPU/MPS. On a CUDA box
    running the 8B, pass torch.bfloat16 to fit memory; that is a deliberate
    fidelity/size tradeoff, not the default.

    device_map (e.g. "auto") shards the model across multiple GPUs via accelerate.
    Needed when the model does not fit on one card — e.g. the 8B in bf16 (~16 GB)
    across Kaggle's 2x T4 (16 GB each). When set, we do not call .to(); accelerate
    places the layers, and inputs go to the embedding layer's device. When None,
    the model loads on a single device as usual.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map, revision=revision
        )
        model.eval()
        # Inputs must land on whatever device holds the embedding layer.
        input_device = str(model.get_input_embeddings().weight.device)
        return LoadedModel(
            tokenizer=tokenizer,
            model=model,
            device=input_device,
            n_layers=model.config.num_hidden_layers,
            hidden_size=model.config.hidden_size,
        )

    device = pick_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, revision=revision
    )
    model.to(device)
    model.eval()
    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
        n_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
    )


def template_ids(tokenizer, conv: Conversation) -> list[int]:
    """Token ids for a templated conversation, add_generation_prompt=False."""
    enc = tokenizer.apply_chat_template(
        conv, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    return enc["input_ids"][0].tolist()


def turn_spans(tokenizer, conv: Conversation) -> list[tuple[int, int]]:
    """Token span [start, end) of each message in the templated conversation.

    Computed by templating growing prefixes of the conversation and taking the
    differences in length. That is only valid if the chat template is
    prefix-consistent — templating the first k messages must produce exactly the
    first N tokens of the full sequence. Llama-3 style templates are, because
    each message is independently wrapped in header/eot delimiters, but a
    template that appends anything after the last message (or re-emits a system
    block) would not be, and would silently shift every read index.

    So we verify rather than assume, and raise if it does not hold. A wrong span
    here would corrupt every multi-turn result while looking perfectly healthy.
    """
    full = template_ids(tokenizer, conv)
    bounds = [0]
    for k in range(1, len(conv) + 1):
        prefix = template_ids(tokenizer, conv[:k])
        if prefix != full[: len(prefix)]:
            chatml = _chatml_turn_spans(tokenizer, conv, full)
            if chatml is not None:
                return chatml
            raise ValueError(
                "chat template is not prefix-consistent and no validated "
                "message-delimiter fallback applies; a read-point implementation "
                "specific to this template is needed"
            )
        bounds.append(len(prefix))
    return list(zip(bounds[:-1], bounds[1:]))


def _chatml_turn_spans(
    tokenizer, conv: Conversation, full: list[int]
) -> list[tuple[int, int]] | None:
    """Validated fallback for ChatML templates such as Qwen3.

    Qwen3 inserts an empty ``<think>`` block when an assistant message is the
    final message in a rendered prefix, but omits it when that same assistant
    message is followed by another user turn. Growing rendered prefixes are
    therefore not token prefixes of the full conversation. The full template
    still has one ``<|im_start|> ... <|im_end|>`` segment per input message, so
    those explicit boundaries are safer than trying to compensate for the
    thinking block.

    This fallback is deliberately narrow: both delimiter tokens must be in the
    tokenizer vocabulary, there must be exactly one start and one end marker
    per input message, and every decoded segment must begin with its expected
    role header. Otherwise it declines and the caller raises.
    """
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        return None
    vocab = get_vocab()
    start_token, end_token = "<|im_start|>", "<|im_end|>"
    if start_token not in vocab or end_token not in vocab:
        return None
    start_id, end_id = vocab[start_token], vocab[end_token]
    starts = [index for index, token in enumerate(full) if token == start_id]
    ends = [index for index, token in enumerate(full) if token == end_id]
    if len(starts) != len(conv) or len(ends) != len(conv):
        return None
    boundaries = starts + [len(full)]
    spans = list(zip(boundaries[:-1], boundaries[1:]))
    for message, (start, end), marker in zip(conv, spans, ends):
        if not start <= marker < end:
            return None
        segment = tokenizer.decode(full[start:end], skip_special_tokens=False)
        if not segment.startswith(f"{start_token}{message['role']}\n"):
            return None
    return spans


def _last_index_of_role(conv: Conversation, spans, roles: tuple[str, ...]) -> int:
    for i in range(len(conv) - 1, -1, -1):
        if conv[i]["role"] in roles:
            return spans[i][1] - 1
    raise ValueError(f"conversation has no message with role in {roles}")


def read_slice(read_point: str, conv: Conversation, spans) -> tuple[int, int]:
    """Token span [start, end) to pool over for a single-vector read point.

    A one-token read point returns a width-1 span, so the caller can treat all
    read points uniformly by mean-pooling over the returned slice.
    """
    if read_point == "last_user_token":
        i = _last_index_of_role(conv, spans, ("user",))
        return i, i + 1
    if read_point == "post_instruction_token":
        if conv and conv[0]["role"] == "system":
            i = spans[0][1] - 1
        else:
            first_user = next(
                (k for k, m in enumerate(conv) if m["role"] == "user"), None
            )
            if first_user is None:
                raise ValueError("conversation has no user message")
            i = spans[first_user][1] - 1
        return i, i + 1
    if read_point == "final_turn_mean":
        for i in range(len(conv) - 1, -1, -1):
            if conv[i]["role"] == "user":
                return spans[i]
        raise ValueError("conversation has no user message")
    if read_point == "final_assistant_token":
        i = _last_index_of_role(conv, spans, ("assistant",))
        return i, i + 1
    if read_point == "final_assistant_mean":
        for i in range(len(conv) - 1, -1, -1):
            if conv[i]["role"] == "assistant":
                return spans[i]
        raise ValueError("conversation has no assistant message")
    raise ValueError(f"unknown read_point {read_point!r}; known: {READ_POINTS}")


@torch.no_grad()
def extract_activations(
    lm: LoadedModel,
    conversations: list[Conversation],
    read_point: str = "last_user_token",
    show_progress: bool = True,
) -> np.ndarray:
    """Extract residual-stream activations at one token position, every layer.

    Returns an array of shape (n_conversations, n_layers + 1, hidden_size),
    float32 on CPU. Axis 1 index 0 is the embedding layer; indices 1..n_layers
    are transformer block outputs.
    """
    known = READ_POINTS + ASSISTANT_READ_POINTS
    if read_point not in known:
        raise ValueError(f"unknown read_point {read_point!r}; known: {known}")

    single_turn = read_point == "last_user_token"
    out = np.empty(
        (len(conversations), lm.n_layers + 1, lm.hidden_size), dtype=np.float32
    )
    iterator = tqdm(conversations, disable=not show_progress, desc="extract")
    for i, conv in enumerate(iterator):
        enc = lm.tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        seq_len = enc["input_ids"].shape[1]
        if single_turn and conv and conv[-1]["role"] == "user":
            # Fast path, and the slice-1 behaviour exactly: the last token of a
            # conversation ending in a user turn is that turn's closing eot.
            start, end = seq_len - 1, seq_len
        else:
            spans = turn_spans(lm.tokenizer, conv)
            start, end = read_slice(read_point, conv, spans)

        enc = {k: v.to(lm.device) for k, v in enc.items()}
        result = lm.model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (n_layers + 1) tensors, each (1, seq, hidden).
        # Mean-pool the read span per layer -> (n_layers + 1, hidden). A
        # single-token read point has a width-1 span, so this is a no-op there.
        vecs = [
            hs[0, start:end, :].float().mean(dim=0).cpu().numpy()
            for hs in result.hidden_states
        ]
        out[i] = np.stack(vecs, axis=0)
    return out


@torch.no_grad()
def extract_trajectories(
    lm: LoadedModel,
    conversations: list[Conversation],
    show_progress: bool = True,
) -> list[np.ndarray]:
    """One activation vector per user turn, to locate where the signal emerges.

    Returns a list (one entry per conversation) of arrays shaped
    (n_user_turns, n_layers + 1, hidden_size). Turn counts vary, so this is a
    list rather than one array; probe code should either fix the turn index or
    interpolate onto a common grid before averaging across conversations.

    Each turn's vector is read at that turn's final token, so turn k's vector
    reflects everything up to and including turn k and nothing after it.
    """
    out: list[np.ndarray] = []
    iterator = tqdm(conversations, disable=not show_progress, desc="trajectories")
    for conv in iterator:
        spans = turn_spans(lm.tokenizer, conv)
        idx = [spans[i][1] - 1 for i, m in enumerate(conv) if m["role"] == "user"]
        if not idx:
            raise ValueError("conversation has no user message")
        enc = lm.tokenizer.apply_chat_template(
            conv, add_generation_prompt=False, return_dict=True, return_tensors="pt"
        )
        enc = {k: v.to(lm.device) for k, v in enc.items()}
        result = lm.model(**enc, output_hidden_states=True)
        per_layer = [hs[0].float().cpu().numpy() for hs in result.hidden_states]
        # (n_user_turns, n_layers + 1, hidden)
        out.append(np.stack([np.stack([pl[j] for pl in per_layer]) for j in idx]))
    return out


@torch.no_grad()
def extract_trajectories_prefixwise(
    lm: LoadedModel,
    conversations: list[Conversation],
    show_progress: bool = True,
) -> list[np.ndarray]:
    """Extract each user-turn state from an independent causal-prefix pass.

    This is more expensive than ``extract_trajectories`` but avoids a subtle
    cross-template/kernel assumption. Some models choose different BF16
    attention kernels for different total sequence lengths. Even with a causal
    mask, that can produce suffix-length-dependent numerical differences at an
    earlier token. Running the exact prefix through the user boundary makes
    the tensor shape, tokens, and causal context identical for matched-prefix
    controls, so a pre-final difference is again a hard implementation check.
    """
    out: list[np.ndarray] = []
    iterator = tqdm(conversations, disable=not show_progress,
                    desc="prefix trajectories")
    for conv in iterator:
        user_ends = [index for index, message in enumerate(conv)
                     if message["role"] == "user"]
        if not user_ends:
            raise ValueError("conversation has no user message")
        item = []
        for message_index in user_ends:
            prefix = conv[: message_index + 1]
            enc = lm.tokenizer.apply_chat_template(
                prefix,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            enc = {key: value.to(lm.device) for key, value in enc.items()}
            result = lm.model(**enc, output_hidden_states=True)
            item.append(np.stack([
                hidden[0, -1, :].float().cpu().numpy()
                for hidden in result.hidden_states
            ]))
        out.append(np.stack(item))
    return out
