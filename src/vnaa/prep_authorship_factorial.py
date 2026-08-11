"""Freeze probe-blind item selection for the authorship-factorial pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = Path("datasets/slice3/coherent_pairs/validated.json")
PRESPEC = Path("datasets/authorship/prespec.md")
MANIFEST = Path("datasets/authorship/pilot_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_two_turn_items(rows: list[dict], n: int) -> list[str]:
    """Select by item-id hash only, after mechanical paired-design checks."""
    eligible = []
    seen = set()
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("every row needs a non-empty string item_id")
        if item_id in seen:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen.add(item_id)
        eval_conv, deploy_conv = row["eval"], row["deploy"]
        if eval_conv[:-1] != deploy_conv[:-1]:
            raise ValueError(f"{item_id} does not have a byte-identical prefix")
        if eval_conv[-1]["role"] != "user" or deploy_conv[-1]["role"] != "user":
            raise ValueError(f"{item_id} does not end in a user turn")
        user_turns = sum(message["role"] == "user" for message in eval_conv)
        if user_turns == 2 and len(eval_conv) == 3:
            eligible.append(item_id)
    if len(eligible) < n:
        raise ValueError(f"need {n} eligible two-turn pairs, found {len(eligible)}")
    eligible.sort(key=lambda item_id: hashlib.sha256(item_id.encode("utf-8")).hexdigest())
    return eligible[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--prespec", type=Path, default=PRESPEC)
    parser.add_argument("--output", type=Path, default=MANIFEST)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    rows = json.load(open(args.source, encoding="utf-8"))["items"]
    selected = select_two_turn_items(rows, args.n)
    manifest = {
        "status": "frozen_before_generation",
        "selection_rule": "first n eligible two-user-turn item_ids ordered by sha256(item_id)",
        "n": args.n,
        "source": args.source.as_posix(),
        "source_sha256": sha256_file(args.source),
        "prespec": args.prespec.as_posix(),
        "prespec_sha256": sha256_file(args.prespec),
        "item_ids": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"saved {args.output} ({len(selected)} item ids)")
    print(f"source_sha256={manifest['source_sha256']}")
    print(f"prespec_sha256={manifest['prespec_sha256']}")


if __name__ == "__main__":
    main()
