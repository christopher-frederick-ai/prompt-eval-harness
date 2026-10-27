"""Experiment inputs and prompt variants.

The inputs are short synthetic paragraphs written for this project (varied domain and
length, including one very short and one long edge case). Swap in your own corpus with
`--inputs path/to/file.jsonl` (one {"id": ..., "text": ...} object per line).
"""

from __future__ import annotations

import json
from pathlib import Path

INPUTS: list[dict] = [
    {
        "id": "p01",
        "domain": "ai-ops",
        "text": (
            "Observability for AI agents means recording every step a request takes: the prompt, "
            "each tool call, the model's reply, and the time and tokens each step used. With those "
            "traces, a team can replay a failed session, find the step that went wrong, and tell a "
            "slow tool from a slow model. Without them, a wrong answer is just a mystery."
        ),
    },
    {
        "id": "p02",
        "domain": "environment",
        "text": (
            "Coral reefs bleach when ocean temperatures stay above normal for several weeks. The "
            "stressed coral expels the algae that live in its tissue and give it both colour and "
            "food. Bleached coral is not dead, and it can recover if conditions cool quickly, but "
            "repeated events leave too little time between bouts for reefs to rebuild."
        ),
    },
    {
        "id": "p03",
        "domain": "urban-policy",
        "text": (
            "A mid-sized city ran a two-year pilot of protected bike lanes on four main roads. "
            "Bike trips on those roads rose by about 40 percent, while car travel times increased "
            "by less than a minute at peak hours. Shop owners were worried at first, but sales on "
            "the pilot streets were flat or slightly higher by the end of the second year."
        ),
    },
    {
        "id": "p04",
        "domain": "history",
        "text": (
            "Before movable-type printing spread across Europe in the fifteenth century, books were "
            "copied by hand and were expensive and rare. Printing cut the cost of a book "
            "dramatically and made it possible to produce identical copies, which helped scientific "
            "findings, religious arguments and everyday news travel faster and further than before."
        ),
    },
    {
        "id": "p05",
        "domain": "personal-finance",
        "text": (
            "Compound interest means earning returns on both the money you put in and the returns "
            "already earned. Over short periods the effect looks small, but over decades it "
            "dominates. This is why starting to save early matters more than saving a larger amount "
            "later, and why fees that look tiny each year can consume a large share of long-run gains."
        ),
    },
    {
        "id": "p06",
        "domain": "logistics",
        "text": (
            "When a major port becomes congested, ships wait offshore, containers pile up, and "
            "trucking schedules slip. Retailers respond by ordering earlier and holding more "
            "inventory, which ties up cash and warehouse space. Delays at one port also push cargo "
            "to alternatives that were never built to absorb the extra volume, spreading the jam."
        ),
    },
    {
        "id": "p07",
        "domain": "urban-science",
        "text": (
            "Dense city centres are often several degrees warmer than the surrounding countryside, "
            "an effect known as the urban heat island. Dark roofs and pavement absorb sunlight, "
            "buildings block wind, and there are fewer plants to cool the air by evaporation. "
            "Planting trees, using lighter surfaces and adding green roofs can reduce the difference."
        ),
    },
    {
        "id": "p08",
        "domain": "software",
        "text": (
            "Technical debt describes the future cost of choosing a quick solution now. A shortcut "
            "can be the right call when a deadline is real, but the debt accrues interest: every "
            "later change takes longer because the code is harder to understand. Teams that track "
            "their debt and schedule time to repay it avoid a slowdown that otherwise arrives "
            "gradually and then all at once."
        ),
    },
    {
        "id": "p09",
        "domain": "economics",
        "text": (
            "Inflation reduces what a given amount of money can buy. If wages rise more slowly than "
            "prices, households feel poorer even though their paycheck is larger in nominal terms. "
            "Economists therefore compare real wages, which adjust for price changes, rather than "
            "nominal wages when judging whether living standards are improving."
        ),
    },
    {
        "id": "p10",
        "domain": "space",
        "text": (
            "Radio signals take between roughly four and twenty-four minutes to travel one way "
            "between Earth and Mars, depending on where the planets are in their orbits. Rovers "
            "therefore cannot be driven in real time. Operators send a plan for the day, the rover "
            "carries it out on its own, and the results are reviewed the next morning."
        ),
    },
    {
        "id": "p11",
        "domain": "short-edge-case",
        "text": "The meeting has moved to Thursday at 3 pm. Please bring the updated budget.",
    },
    {
        "id": "p12",
        "domain": "long-edge-case",
        "text": (
            "Remote work changed more than where people sit. Teams that once relied on hallway "
            "conversations had to write things down, and many discovered that clear written "
            "decisions made meetings shorter and onboarding faster. At the same time, managers "
            "found it harder to notice when someone was struggling, and newer employees missed the "
            "informal mentoring that happens when people share a room. Companies that did well "
            "tended to be explicit about which work needs real-time collaboration and which does "
            "not, to set norms for response times, and to bring people together in person for "
            "planning and relationship-building rather than for routine updates. Those that treated "
            "remote work as the old office with the video turned on often saw communication "
            "overhead grow without any of the flexibility benefits."
        ),
    },
]

# Prompt prefixes. Text is appended directly after the prefix. v1-v4 match the course demo;
# v5 tests the hypothesis from the pilot that adding a constraint to TL;DR fixes its cost.
VARIANTS: dict[str, str] = {
    "v1": "Summarize the following in one sentence:\n",
    "v2": "Write a short summary of this paragraph:\n",
    "v3": "Condense this into a concise statement:\n",
    "v4": "TL;DR:\n",
    "v5": "TL;DR in one sentence:\n",
    # Multi-sentence instruction: tests whether a longer, more specific prompt (more INPUT
    # tokens) buys consistency or quality, and whether that is worth its extra input cost.
    "v6": (
        "You are a careful editor. Summarize the paragraph below in a single sentence of at most "
        "25 words. Keep the key facts, do not add anything that is not in the paragraph, and "
        "return only the summary.\n\n"
    ),
}

# Variants whose prompt explicitly demands a single sentence (used for the adherence check).
ONE_SENTENCE_VARIANTS = {"v1", "v5", "v6"}

# Variants that state an explicit word limit (used for the word-limit adherence check).
WORD_LIMITS = {"v6": 25}

BASELINE_VARIANT = "v1"


def build_prompt(variant_id: str, text: str) -> str:
    return VARIANTS[variant_id] + text


def load_inputs(path: str | Path | None = None) -> list[dict]:
    """Built-in inputs, or a JSONL file of {"id", "text"} objects."""
    if path is None:
        return list(INPUTS)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                items.append({"id": str(obj["id"]), "domain": obj.get("domain", ""), "text": obj["text"]})
    if len(items) < 3:
        raise ValueError("Need at least 3 inputs for meaningful statistics.")
    return items
