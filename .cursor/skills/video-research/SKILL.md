---
name: video-research
description: >-
  Runs comprehensive YouTube video research via the plan4me MCP before deep
  topic work. Searches talks/interviews, pulls transcripts, extracts cited
  knowledge atoms, and synthesizes a Markdown guide. Use when the user wants
  research that needs watching or reading many videos, interviews, talks,
  podcasts, or YouTube sources; when building evidence-backed guides from
  video; or before a comprehensive research pass on a topic with rich video
  coverage.
---

# Video Research (plan4me MCP)

## When to use

Use this skill **before** comprehensive research when the topic needs many
YouTube interviews, talks, or podcasts — not for docs-only or code-only tasks.

MCP server name: **plan4me**

## Tools

| Tool | Use |
|------|-----|
| `health` | Confirm Bedrock/credentials before a long run |
| `search_videos` | Cheap preview of YouTube candidates |
| `research_topic` | Full pipeline (minutes). Returns cited Markdown |
| `get_latest_report` | Recover last saved report without re-running |

## Workflow

Copy and track:

```
Video research:
- [ ] 1. Clarify topic + angles
- [ ] 2. health (optional)
- [ ] 3. search_videos for each angle
- [ ] 4. research_topic per angle (or one broad topic)
- [ ] 5. Synthesize across reports for the user
```

### Single-topic research

1. Call `search_videos` with the topic (`max_results` 10–20) to sanity-check hits.
2. Call `research_topic` with a clear topic string and `max_videos` 8–15 (default 10).
3. Return the Markdown guide; cite video titles from the report. Do not invent sources.

### Multi-angle research (subagents)

When the topic spans distinct angles (e.g. pricing, GTM, hiring):

1. Split into 2–4 focused queries.
2. Launch parallel **Task** subagents (or the `video-researcher` agent), each calling `research_topic` for one angle.
3. Merge Key Takeaways, Frameworks, Contradictions, and Actionable Checklists into one answer.
4. Prefer claims that appear across multiple videos/angles.

### research_topic tips

- Phrase `topic` like a YouTube search for expert talks, not a vague essay title.
- `languages`: comma-separated codes (e.g. `en` or `en,hi`) when needed.
- Expect several minutes; do not re-call on timeout without checking `get_latest_report`.
- Higher `max_videos` → better coverage, more time/cost.

## Output for the user

Lead with the synthesized guide (or merged multi-angle summary). Keep provenance:
video titles and timestamps from the report. Flag contradictions explicitly.
