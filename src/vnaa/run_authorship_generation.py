"""Generate the frozen history-authorship pilot, one generator at a time.

Each invocation loads exactly one unquantized model and writes an atomic,
resumable result file.  The generated assistant history is later crossed with
both frozen final-user endings; generation never sees either ending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file

from vnaa.prep_authorship_factorial import sha256_file


SOURCE = Path("datasets/slice3/coherent_pairs/validated.json")
MANIFEST = Path("datasets/authorship/pilot_manifest.json")
OUTPUT_DIR = Path("results/authorship/generations")

GENERATORS = {
    "target_on_policy": "meta-llama/Llama-3.1-8B-Instruct",
    "same_family_off_policy": "meta-llama/Llama-3.2-3B-Instruct",
    "cross_family_off_policy": "Qwen/Qwen2.5-7B-Instruct",
}


def sha256_token_ids(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def termination_ids(tokenizer, model) -> list[int]:
    """Return recognized EOS/end-of-turn ids for this native chat template."""
    ids: set[int] = set()
    for source in (tokenizer, model.config, model.generation_config):
        value = getattr(source, "eos_token_id", None)
        if isinstance(value, int):
            ids.add(value)
        elif value is not None:
            ids.update(int(x) for x in value)
    for token in ("<|eot_id|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    if not ids:
        raise ValueError("tokenizer/model exposes no recognized termination token")
    return sorted(ids)


def load_frozen_inputs(source: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(source) != manifest["source_sha256"]:
        raise ValueError("source hash no longer matches the frozen pilot manifest")
    prespec = Path(manifest["prespec"])
    if sha256_file(prespec) != manifest["prespec_sha256"]:
        raise ValueError("prespec hash no longer matches the frozen pilot manifest")

    source_rows = json.loads(source.read_text(encoding="utf-8"))["items"]
    by_id = {row["item_id"]: row for row in source_rows}
    rows = []
    for item_id in manifest["item_ids"]:
        row = by_id[item_id]
        eval_conv, deploy_conv = row["eval"], row["deploy"]
        if len(eval_conv) != 3 or eval_conv[:-1] != deploy_conv[:-1]:
            raise ValueError(f"{item_id}: frozen source is not the required paired design")
        rows.append({
            "item_id": item_id,
            "generation_messages": [eval_conv[0]],
            "eval_final": eval_conv[-1],
            "deploy_final": deploy_conv[-1],
        })
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolved_commit(obj: Any) -> str:
    commit = getattr(obj, "_commit_hash", None)
    if commit is None and hasattr(obj, "init_kwargs"):
        commit = obj.init_kwargs.get("_commit_hash")
    if not isinstance(commit, str) or len(commit) < 7:
        raise ValueError("could not recover the model/tokenizer resolved revision")
    return commit


def resolved_repo_commit(repository: str, filename: str) -> str:
    """Recover the immutable Hub snapshot revision used for one artifact."""
    resolved = Path(cached_file(repository, filename))
    parts = resolved.parts
    try:
        snapshot_index = parts.index("snapshots")
        commit = parts[snapshot_index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"could not recover snapshot revision for {repository}/{filename}") from error
    if len(commit) < 7:
        raise ValueError(f"invalid snapshot revision for {repository}/{filename}: {commit!r}")
    return commit


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=sorted(GENERATORS), required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model_name = GENERATORS[args.condition]
    output = args.output_dir / f"{args.condition}.json"
    inputs = load_frozen_inputs(args.source, args.manifest)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    eos_ids = termination_ids(tokenizer, model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else eos_ids[0]

    metadata = {
        "condition": args.condition,
        "model_repository": model_name,
        "model_revision": resolved_commit(model.config),
        "tokenizer_revision": resolved_repo_commit(model_name, "tokenizer_config.json"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "quantized": False,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "termination_token_ids": eos_ids,
        "source": args.source.as_posix(),
        "source_sha256": sha256_file(args.source),
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": sha256_file(args.manifest),
    }

    saved: dict[str, Any]
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["metadata"] != metadata:
            raise ValueError(f"refusing to resume {output}: metadata differs")
    else:
        saved = {"metadata": metadata, "items": []}
    completed = {row["item_id"] for row in saved["items"]}

    for number, row in enumerate(inputs, start=1):
        if row["item_id"] in completed:
            continue
        rendered = tokenizer.apply_chat_template(
            row["generation_messages"],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_ids = rendered["input_ids"][0].tolist()
        rendered = {key: value.to(model.device) for key, value in rendered.items()}
        generated = model.generate(
            **rendered,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
        )[0, len(prompt_ids):].tolist()
        terminated = bool(generated and generated[-1] in eos_ids)
        response_ids = generated[:-1] if terminated else generated
        response = tokenizer.decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        saved["items"].append({
            "item_id": row["item_id"],
            "rendered_input_sha256": sha256_token_ids(prompt_ids),
            "rendered_input_token_ids": prompt_ids,
            "output_token_ids": generated,
            "response": response,
            "terminated_naturally": terminated,
            "termination_token_id": generated[-1] if terminated else None,
            "generation_length": len(generated),
        })
        atomic_json(output, saved)
        print(f"[{number:02d}/{len(inputs)}] {row['item_id']} tokens={len(generated)} "
              f"terminated={terminated}", flush=True)

    print(f"saved {output} ({len(saved['items'])} items)")


if __name__ == "__main__":
    main()
