"""Figure 1: how the two paired temporal controls are constructed.

Both controls hold the conversation history fixed within a pair and vary only
the final user turn, so the pre-final read point sees byte-identical context
under both labels. They differ in where that history comes from and in how far
the two endings are matched.
"""

from __future__ import annotations

from matplotlib.patches import FancyArrowPatch

import figstyle
from figstyle import (
    BLUE,
    MODEL_FILL,
    MUTED,
    NEUTRAL,
    ORANGE,
    TURN_GAP,
    TURN_HEIGHT,
    TURN_WIDTH,
    USER_FILL,
    read_marker,
    span_bracket,
    turn_box,
)

# figstyle selects a non-interactive backend on import.
import matplotlib.pyplot as plt


def draw_control(
    axis,
    letter,
    title,
    prefix_label,
    prefix_turns,
    eval_label,
    deploy_label,
    ending_note,
) -> None:
    baseline = 0.0
    branch_offset = 0.52

    x = 0.0
    for index, name in enumerate(prefix_turns):
        fill = USER_FILL if index % 2 == 0 else MODEL_FILL
        turn_box(axis, x, baseline, name, fill)
        x += TURN_WIDTH + TURN_GAP

    prefix_end = x - TURN_GAP
    span_bracket(axis, 0.0, prefix_end, TURN_HEIGHT / 2 + 0.10, prefix_label)
    read_marker(
        axis,
        prefix_end - TURN_WIDTH / 2,
        -TURN_HEIGHT / 2 - 0.05,
        "pre-final\nread",
    )

    # The pair diverges only here: one final user turn per label.
    branch_x = x + 0.34
    ending_width = 2.35
    for offset, label, color in (
        (branch_offset, eval_label, ORANGE),
        (-branch_offset, deploy_label, BLUE),
    ):
        axis.add_patch(
            FancyArrowPatch(
                (prefix_end + 0.04, baseline),
                (branch_x - 0.04, baseline + offset),
                arrowstyle="-|>",
                mutation_scale=6,
                color=MUTED,
                linewidth=0.7,
                shrinkA=0,
                shrinkB=0,
            )
        )
        turn_box(
            axis,
            branch_x,
            baseline + offset,
            label,
            "white",
            edge=color,
            text_color=color,
            width=ending_width,
            fontsize=6.2,
        )

    read_marker(
        axis,
        branch_x + ending_width / 2,
        baseline - branch_offset - TURN_HEIGHT / 2 - 0.05,
        "final read",
    )
    axis.text(
        branch_x + ending_width + 0.22,
        baseline,
        ending_note,
        ha="left",
        va="center",
        fontsize=6.0,
        color=NEUTRAL,
        linespacing=1.35,
    )

    axis.set_xlim(-0.12, branch_x + ending_width + 2.85)
    axis.set_ylim(-1.06, 0.86)
    axis.set_aspect("equal")
    axis.axis("off")
    figstyle.panel_label(axis, letter, title, pad=2.0)


def main() -> None:
    figstyle.use_paper_style()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figstyle.TEXT_WIDTH_IN, 2.55),
        gridspec_kw={"hspace": 0.22},
    )

    draw_control(
        axes[0],
        "a",
        "Retained-history control",
        "byte-identical within pair",
        ["User 1", "Assistant 1", "User 2", "Assistant 2"],
        "casually rendered\nbenchmark item",
        "original final\nuser message",
        "Real WildChat history.\nThe replacement ending may\nshift topic and speech act.",
    )
    draw_control(
        axes[1],
        "b",
        "Coherence-matched control",
        "byte-identical within pair",
        ["User 1", "Assistant 1", "User 2", "Assistant 2"],
        "faithful casual\nbenchmark question",
        "genuine practical request,\nsame topic",
        "History and endings\nconstructed together.\nEndings matched on topic,\nregister and length.",
    )

    handles = [
        plt.Line2D([], [], color=ORANGE, marker="s", linestyle="none", markersize=4.5),
        plt.Line2D([], [], color=BLUE, marker="s", linestyle="none", markersize=4.5),
    ]
    fig.legend(
        handles,
        ["Evaluation ending", "Deployment ending"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
    )

    figstyle.save(fig, "paired_controls")


if __name__ == "__main__":
    main()
