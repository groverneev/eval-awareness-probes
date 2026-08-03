"""Slice 2b: prepare rewrite batches.

Design settled after the prototype gate (see CLAUDE.md, slice 2b notes):

1. Claude-authored text is surface-separable from real ShareGPT text (format
   0.93, BoW 0.91) even when its CONTENT is deployment. So every cell in the
   decisive contrast is Claude-authored: eval items and deploy items both get
   rewritten. Authorship then cancels instead of confounding.

2. Two eval cells, not one:
   - style A: the benchmark question asked plainly and casually.
   - style B: the same question wrapped in a plausible personal need.
   Reporting both separates "quiz flavour" from genuine content recognition.

3. Surface matching is built into the inputs rather than hoped for:
   - each item carries an explicit target length; the same multiset of target
     lengths is dealt to all three cells, so length distributions match by
     construction rather than by post-hoc histogram matching.
   - the ShareGPT source sample is chosen to match the benchmark sample's
     digit-density distribution, because benchmark maths items carry numbers
     and generic user questions do not (digit density was the single worst
     surface feature in the prototype, AUROC 0.69).

4. One writer handles a whole batch — its eval items AND its deploy items — so
   per-writer style variation is balanced across cells instead of correlating
   with the label.

Outputs one JSON file per batch under datasets/casual_eval_true/batches/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from vnaa.build_2b import OUT_DIR, read_jsonl
from vnaa.data_2x2 import load_cell

BATCH_DIR = OUT_DIR / "batches"

# MMLU in this dump contains items whose stem references a numbered list that
# was never included ("They are too irrational and uncodified." with choices
# "3,4" / "1,3"). They are unanswerable standalone and cannot be rewritten.
_LIST_CHOICE = re.compile(r"^[\d\s,]+$")


# Items whose stem does not pin the answer without its option list. A writer
# asked to rewrite one of these has to fold the options back into prose ("a
# fusion crust, spherical droplets, ... which of those doesn't mean it is a
# meteorite?"), which re-introduces the enumerative MCQ shape the whole arm
# exists to remove. Cheaper to drop them.
_NEEDS_OPTIONS = re.compile(
    r"(which of the following|which of these|all of the following|"
    r"statement 1|which one of|following is (not |never |always )?(true|correct|false))",
    re.I,
)


def is_malformed(item: dict) -> bool:
    q = item["question"].strip()
    if len(q) < 25:
        return True
    choices = item["choices"]
    if choices and all(_LIST_CHOICE.match(str(c)) for c in choices):
        return True
    if _NEEDS_OPTIONS.search(q):
        return True
    return False


def digit_density(text: str) -> float:
    return sum(c.isdigit() for c in text) / max(len(text), 1)


# Real user traffic contains homework: ~16% of the ShareGPT prompts in the
# first sampled batch were math word problems ("A water tank can be filled by
# two pipes in 11 1/9 minutes..."), which are indistinguishable in content from
# GSM8K eval items. Worse, matching on digit density actively SELECTS for them,
# because that is where the digits are. Such prompts are label noise in the
# deployment cell, so they are filtered from the pool before matching.
_QUIZ_CUE = re.compile(
    r"(how many|how much|what is the (value|price|total|ratio|area|sum)|"
    r"calculate |find the |solve |prove that|simplify |evaluate the)",
    re.I,
)


def looks_like_quiz(text: str) -> bool:
    """Deployment prompts that are really test questions in disguise."""
    return bool(_QUIZ_CUE.search(text)) and bool(re.search(r"\d", text))


def is_question_form(text: str) -> bool:
    """Does this prompt ask something, rather than command the assistant?

    The first pilot gate failed mainly on this axis: 100% of eval items
    contained a question mark vs 54% of deploy items, because benchmark items
    are questions while user prompts are often imperatives ("write me a post").
    Real traffic contains plenty of genuine questions, so we can select for
    them and balance the axis without touching the labels.
    """
    t = text.strip().lower()
    if "?" in t:
        return True
    return bool(re.match(r"(what|why|how|when|where|which|who|is |are |can |does|do |should|could|would)\b", t))


def match_pool(
    pool: list[str],
    items: list[dict],
    rng: np.random.Generator,
    digit_weight: float = 0.5,
) -> list[str]:
    """Pick one real deploy prompt per eval item: topically close, digit-matched.

    Greedy nearest-neighbour without replacement over a combined cost of
    (1 - tfidf cosine similarity) and |digit density difference|. Topic matching
    is what makes the bag-of-words baseline meaningful: without it, BoW just
    reads "science words vs coding words" and reaches 0.93 whatever the probe
    does. Matching against REAL prompts (rather than writing synthetic
    counterparts) keeps the deployment side derived from real traffic.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    questions = [it["question"] for it in items]
    vec = TfidfVectorizer(stop_words="english", min_df=1)
    mat = vec.fit_transform(pool + questions)
    pool_m, item_m = mat[: len(pool)], mat[len(pool):]
    sim = (item_m @ pool_m.T).toarray()  # (n_items, n_pool)

    pool_dens = np.array([digit_density(t) for t in pool])
    item_dens = np.array([digit_density(q) for q in questions])
    cost = (1.0 - sim) + digit_weight * np.abs(
        item_dens[:, None] - pool_dens[None, :]
    ) / max(pool_dens.std(), 1e-6)

    used = np.zeros(len(pool), dtype=bool)
    out = []
    # Hardest-to-match items first, so they are not left with scraps.
    order = np.argsort(-cost.min(axis=1))
    picks: dict[int, str] = {}
    for i in order:
        c = np.where(used, np.inf, cost[i])
        j = int(np.argmin(c))
        used[j] = True
        picks[int(i)] = pool[j]
    out = [picks[i] for i in range(len(items))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare slice 2b rewrite batches")
    ap.add_argument("--n", type=int, default=300, help="items per cell")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    items = read_jsonl(OUT_DIR / "bench_eval_sample.jsonl")
    kept = [it for it in items if not is_malformed(it)]
    print(f"bench sample: {len(items)} -> {len(kept)} after dropping malformed")
    if len(kept) < args.n:
        raise SystemExit(f"only {len(kept)} usable items, need {args.n}")
    kept = [kept[i] for i in rng.permutation(len(kept))[: args.n]]

    # Target lengths are derived from each benchmark question's OWN length, not
    # from the ShareGPT length distribution. The first pilot batch drew targets
    # from real prompts (mean 193 chars) — far longer than a bare ARC question —
    # so the writer padded style A with personal framing to reach the target and
    # style A collapsed into style B (44% of style-A items carried first-person
    # pronouns). Targets sized to the question keep style A naturally terse; the
    # same multiset is dealt to all cells, so lengths still match by design.
    q_lens = np.array([len(it["question"]) for it in kept], dtype=float)
    targets = np.clip(np.round(q_lens * 0.95), 70, 200).astype(int)

    # Realistic deploy source: WildChat first turns. Measured against ShareGPT,
    # WildChat matches our benchmark items marginally better (mean best-match
    # tfidf similarity 0.127 vs 0.104, BoW after matching 0.935 vs 0.951) and
    # slice 3 needs it anyway. Neither source solves the topic gap — real users
    # do not ask about xylem cells — which is why a synthetic topic-matched
    # deploy cell is built alongside it as the clean control.
    wc_path = Path("data/wildchat_first_turns.json")
    if not wc_path.exists():
        raise SystemExit(f"missing {wc_path}; see CLAUDE.md for the WildChat sampling snippet")
    with open(wc_path, encoding="utf-8") as f:
        pool = json.load(f)
    pool = [t for t in pool if 40 < len(t) < 600]
    pool = [t for t in pool if not looks_like_quiz(t) and is_question_form(t)]
    print(f"wildchat pool after filtering: {len(pool)}")
    if len(pool) < args.n:
        raise SystemExit(f"only {len(pool)} usable deploy prompts, need {args.n}")

    # One real deploy prompt per eval item, topically closest and digit-matched.
    deploy_src = match_pool(pool, kept, rng)

    perm_eval = rng.permutation(args.n)
    perm_dep = rng.permutation(args.n)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.json"):
        old.unlink()

    n_batches = (args.n + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        lo, hi = b * args.batch_size, min((b + 1) * args.batch_size, args.n)
        eval_items = []
        for k in range(lo, hi):
            it = kept[k]
            eval_items.append({
                "uid": it["uid"],
                "source": it["source"],
                "question": it["question"],
                "choices": it["choices"],
                "target_chars": int(targets[perm_eval[k]]),
            })
        deploy_items = []
        for k in range(lo, hi):
            deploy_items.append({
                "uid": f"dep_{k:04d}",
                "original": deploy_src[k],
                "target_chars": int(targets[perm_dep[k]]),
            })
        # Topic-matched synthetic counterparts: one genuine-need message per
        # eval item, on that item's own subject. This is the only construction
        # that can drive bag-of-words toward chance, because real traffic has
        # no topical overlap with school-subject benchmark items.
        topic_items = [
            {"uid": f"tm_{kept[k]['uid']}", "topic_source": kept[k]["question"],
             "target_chars": int(targets[perm_eval[k]])}
            for k in range(lo, hi)
        ]
        path = BATCH_DIR / f"batch_{b:02d}.json"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"batch": b, "eval_items": eval_items,
                       "deploy_items": deploy_items,
                       "topic_items": topic_items}, f, ensure_ascii=False, indent=1)
        print(f"wrote {path}: {len(eval_items)} eval + {len(deploy_items)} deploy "
              f"+ {len(topic_items)} topic-matched")

    print(f"\ntarget length: mean {targets.mean():.0f} chars, "
          f"range {targets.min()}-{targets.max()}")
    bench_dens = [digit_density(it["question"]) for it in kept]
    print(f"digit density: bench mean {np.mean(bench_dens):.4f}, "
          f"matched deploy mean {np.mean([digit_density(t) for t in deploy_src]):.4f}")


if __name__ == "__main__":
    main()
