"""Final report synthesis from clustered atoms.

The report is generated from the STRUCTURED clusters (with support counts and
citations), not from raw transcripts. This keeps the synthesis faithful and
lets us surface consensus, frequency, and minority/conflicting views.
"""
from __future__ import annotations

import logging

from backend.llm.bedrock import get_synthesis_llm
from backend.schemas import AtomCluster

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a research synthesizer. You are given clusters of knowledge atoms "
    "extracted from many videos about a topic. Each cluster has a support_count "
    "(how many distinct videos expressed it). Produce an exhaustive, well-"
    "structured Markdown knowledge report that PRESERVES information and reads "
    "as one comprehensive guide a learner can follow top to bottom.\n\n"
    "Requirements:\n"
    "- For every point, write a COMPREHENSIVE explanation: not a one-liner. "
    "Explain what the experts actually said, the reasoning, concrete examples, "
    "and how to apply it. Depth and completeness matter more than brevity.\n"
    "- Report consensus with frequencies (e.g. 'mentioned in N sources').\n"
    "- Keep minority and contradicting viewpoints in their own section.\n"
    "- Cite claims using the provided video titles.\n"
    "- Do NOT fabricate anything not supported by the clusters.\n\n"
    "Structure the report with these sections (use '##' headings):\n"
    "1. Overview\n"
    "2. Key Takeaways (a bulleted list of the most important points)\n"
    "3. Detailed Advice (grouped by theme, each point explained comprehensively "
    "with its source count)\n"
    "4. Frameworks & Strategies\n"
    "5. Tools & Resources\n"
    "6. Common Mistakes / Failure Modes\n"
    "7. Contradictions & Debates\n"
    "8. Actionable Checklist\n\n"
    "The 'Actionable Checklist' section MUST use GitHub task-list syntax so each "
    "item is a checkbox, e.g.:\n"
    "- [ ] First concrete action the learner should take\n"
    "- [ ] Second concrete action\n"
    "Make the checklist thorough (10-20 concrete, ordered steps)."
)


def _clusters_to_context(clusters: list[AtomCluster], max_clusters: int = 120) -> str:
    lines: list[str] = []
    for c in clusters[:max_clusters]:
        titles = sorted({a.video_title for a in c.atoms})
        lines.append(
            f"- [{c.type.value}] (sources={c.support_count}) {c.representative_claim}"
        )
        # include a couple of citations + an example action if present
        for a in c.atoms[:3]:
            cite = a.video_title[:70]
            if a.actionable_step:
                lines.append(f"    · action: {a.actionable_step}  (— {cite})")
            elif a.quote:
                lines.append(f'    · "{a.quote}"  (— {cite})')
    return "\n".join(lines)


def synthesize_report(topic: str, clusters: list[AtomCluster]) -> str:
    if not clusters:
        return f"# {topic}\n\nNo knowledge could be extracted for this topic."

    llm = get_synthesis_llm()
    context = _clusters_to_context(clusters)
    prompt = (
        f"TOPIC: {topic}\n\n"
        f"KNOWLEDGE CLUSTERS (sorted by support):\n{context}\n\n"
        "Write the comprehensive Markdown knowledge report now."
    )
    resp = llm.invoke([("system", _SYSTEM), ("human", prompt)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)
