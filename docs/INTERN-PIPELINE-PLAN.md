# Plan for Me — intern implementation plan

You are implementing this inside the repository `plan4me` (`https://github.com/AffanShaikhsurab/plan4me`). On this machine the clone is `C:\Users\affan\Affan Projects\plan4me`.

You have not seen the earlier research chats. This file is the whole assignment. Do not invent a second research method. Do not skip a section because it looks long. When two instructions could be read two ways, the later “Locked decisions” section wins.

Python for this repo is the venv, not system Python:

`C:\Users\affan\Affan Projects\plan4me\.venv\Scripts\python.exe`

The venv is Python 3.12. System Python on this machine is 3.14 and must not be used.

## 0. What you are building

A person types one topic, for example `how to build in public`. Plan for Me watches the top YouTube talks on that topic and returns **one** document a person can follow.

The document is not a summary of each video. The document is an operating procedure: what to do, in order, what not to do, where speakers disagree, and which public pages were actually opened.

There are two depths. Both run on every topic. They are not two products.

| Depth | How many times it runs | What it is allowed to read |
| --- | --- | --- |
| Normal pass | Once per selected video (default 10, hard cap 50) | That video’s captions only |
| Deep pass | Once per topic, after the videos are combined | The public web, for the merged concept list only |

The deep pass does not run once per video. That decision is already made. Do not build 10 or 50 browser-research workers.

## 1. What “done” means

A pull request is done only when all of these are true:

1. `POST /research` with a topic runs the stages in section 8, in that order.
2. Each selected video with captions produces a video brief file. A video with no captions produces a skip record, not a brief invented from the title.
3. One agenda file lists concepts. A concept that appears in only one video is marked `single-source` and is not sent to the deep pass.
4. The deep pass writes one note per concept that the agenda marked `multi-source`, up to 12 concepts. It records every URL it opened.
5. One final Markdown document is saved and returned by `research_topic` and by `GET /research/{job_id}`.
6. Every number in that document has one of the labels in section 6. A speaker’s revenue, follower count, or “this worked for me” is `self-reported`.
7. Two speakers who disagree are both still visible. You did not average them into a third number.
8. No code path downloads a book PDF, searches LibGen, Anna’s Archive, Z-Library, or `filetype:pdf`, or quotes a book the run did not have a licensed file for.
9. Unit tests in section 14 pass with no network. The live check in section 14 was run once on `how to build in public` with `max_videos=3`, and the paths of the output files are in the pull request description.
10. The old five-stage pipeline still runs when `LEGACY_ATOM_PIPELINE=true`. The default after this change is the new pipeline (`LEGACY_ATOM_PIPELINE` unset or `false`).

## 2. The research method this product is copying

Before Plan for Me, research was done by hand inside Cursor, one YouTube video at a time. A parent agent fetched captions. A worker then did all of the following for that single video:

1. Name the topics in the video.
2. Keep the captions, with timestamps. Do not download the video file.
3. Search the public web for those topics.
4. Find the books those talks point at.
5. From lawful summaries of those books, list the detailed concepts. The book itself usually could not be read.
6. Search the public web again for each of those concepts, and open pages that describe steps.
7. Write one research document for that video.

That worker was expensive. Doing it for 50 videos means 50 web researches and 50 documents that still have to be merged by a person.

Plan for Me was supposed to do that worker’s job for the top 10 to 50 videos and then combine them. What shipped is only the caption half, and it combines caption claims with embeddings. It never searches the web and it never maps a book.

The product decision now:

- For each of the top 10 to 50 videos, do only steps 1 and 2. That is the normal pass. Save it.
- Combine those briefs into one list of topics, steps, checklists, tools, named books, and disagreements.
- Run steps 3 through 7 **once** on that combined list. That is the deep pass.
- Write one final document.

## 3. What the code does today

Read these files before you edit them. Naming them is not reading them.

| File | What it does now |
| --- | --- |
| `backend/ingestion/search.py` | `search_videos(query, max_results)` uses yt-dlp `ytsearchN:query` with `extract_flat=True`. Returns `VideoMeta`: `video_id`, `title`, `url`, `channel`, `duration`, `view_count`. No transcript. |
| `backend/ingestion/selection.py` | `select_and_transcribe`. Searches are ranked by YouTube. The first `force_deepgram_top_n` videos (default 5) are always attempted: captions, otherwise Deepgram if `DEEPGRAM_API_KEY` is set. Remaining slots up to `target` are filled by later videos that already have captions. |
| `backend/ingestion/transcripts.py` | Caption fetch. A video with no captions returns nothing. |
| `backend/llm/extraction.py` | Chunks a transcript at 6000 characters. Asks the extraction model for `KnowledgeAtom` items: `type`, `claim`, `actionable_step`, `confidence`, `quote`, `approx_timestamp`. |
| `backend/pipeline/clustering.py` | Embeds each claim. Greedy clustering by cosine similarity. Threshold is `DEDUPE_SIMILARITY_THRESHOLD`, default `0.82`, in `backend/config.py`. `support_count` is the number of distinct `video_id`s in a cluster. |
| `backend/llm/synthesis.py` | Sends up to 120 clusters to the synthesis model. Asks for one Markdown guide with eight fixed sections. |
| `backend/pipeline/graph.py` | Linear graph: `search` → `transcribe` → `extract` → `cluster` → `synthesize`. `run_pipeline(topic, max_videos, languages)`. |
| `backend/pipeline/jobs.py` | `STAGE_ORDER = ("search", "transcribe", "extract", "cluster", "synthesize")`. Background thread. Saves atoms and the report to SQLite. |
| `backend/api/main.py` | `POST /search`, `POST /report` (blocking), `POST /research`, `GET /research/{job_id}`, `GET /reports/latest`. |
| `backend/storage/store.py` | SQLite tables `atoms` and `reports`. File `plan4me.db` at the repo root. Do not commit it. |
| `mcp_server/server.py` | Tools: `health`, `search_videos`, `research_topic`, `get_latest_report`. `research_topic` calls `run_pipeline`. |
| `frontend/app/api.ts` | `ResearchStage` is `search \| transcribe \| extract \| cluster \| synthesize \| done`. |
| `frontend/app/ResearchScene.tsx` | `STAGE_INDEX` maps those stage names to numbers. A new stage name that is missing here will break the typecheck. |
| `backend/schemas.py` | `ReportRequest.max_videos` is `ge=1, le=200`. You will change the cap to 50. |

There is no web search module. There is no book module. Tests under `tests/` cover LLM providers only. There is no pipeline test.

Default models, from `backend/config.py` and `.env.example`: extraction `zai.glm-4.7-flash`, synthesis `deepseek.v3.2`, embeddings `amazon.titan-embed-text-v2:0`, region `us-east-1`, provider `bedrock`. Credentials come from the AWS chain (`~\.aws\credentials` or `AWS_PROFILE`). Do not print keys. Do not put keys in this plan or in a commit.

`NUM_CANDIDATES` defaults to 50. `TARGET_VIDEOS` defaults to 10. A call with `max_videos=10` still searches 50 candidates and keeps up to 10 transcripts.

## 4. What already works

Keep these. Do not rewrite them.

- YouTube search through yt-dlp, and caption fetch through the existing transcript code.
- Skipping a video that has no captions when Deepgram is off. Say that it was skipped. Do not invent a transcript from the title.
- Pluggable chat providers in `backend/llm/providers/`. Call `get_extraction_llm()` for the normal pass and for the agenda merge. Call `get_synthesis_llm()` for the deep pass and the final document. Do not add a new provider.
- Job polling: `POST /research` returns a job id, `GET /research/{job_id}` returns status. Keep that shape. Add fields. Do not replace it with a websocket.
- MCP tool names `health`, `search_videos`, `research_topic`, `get_latest_report`. Agents already call those names.
- Structured output via `backend.llm.chat.structured`. Use it for the video brief and the agenda. Free prose is only allowed for the concept note and the final document.
- The selection rule in `selection.py`: try the top of the search first, then fill with videos that have captions, stop at `max_videos`.

## 5. What does not work, with a measured run

On 14 October 2026 a live run used topic `how to build in public`, `max_videos=10`, languages `en`, Deepgram off.

| Fact | Value |
| --- | --- |
| Candidates searched | 50 |
| Transcripts kept | 10 |
| Skipped for no captions | `6a_erix_tb0`, `3ouqT3lfiUE` |
| Atoms extracted | 247 |
| Clusters | 246 |
| Markdown length | about 22,000 characters |
| Wall time | about 14 minutes |

Clustering did not cluster. 247 claims became 246 clusters, so `support_count` was 1 for almost every claim. The synthesis then wrote “mentioned in 1 source” on ideas that may well be shared. That phrase is not evidence. Do not use cosine clustering as the research agenda. You may leave `cluster_atoms` in the file for the legacy flag. The new pipeline does not call it.

`videos_processed` in `POST /report` is `len(state["videos"])`, and `state["videos"]` is the candidate list from search, not the transcripts. On that run the API would report 50 videos processed when 10 transcripts were used. The new report must separate the two counts. Field names are in section 9.

The guide from that run is a paraphrase of the captions. It names books and templates only when a speaker named them. It did not open those pages. Treat that guide as a demo of the old pipeline, not as a source for the new one.

## 6. Labels

Use these words in stored JSON and in the final document. Do not invent synonyms. Do not promote a label because the sentence sounds confident.

| Label | Use it when |
| --- | --- |
| `verified` | You checked the figure against a primary dataset or an official document, and the note says which document. A YouTube caption is never this. |
| `single-source` | One public source says it. One video, or one web page. |
| `multi-source` | Two or more independent sources say the same action. Two videos count. A video and a web page count. Two pages that rewrite each other do not count. |
| `self-reported` | The speaker is describing their own revenue, followers, close rate, or results. |
| `estimate` | Someone’s rough number, and they say it is rough. |
| `heuristic` | A rule of thumb. Not a measured optimum. “Post every day” is this unless a study was opened. |
| `contested` | Two sources disagree, and this run did not resolve it. |
| `unverified` | A number you cannot tie to a source this run opened. Do not put unverified numbers in “What to do”. Put them in “Not established” or delete them. |
| `summary-derived` | You only know it because summaries of an unread book listed it. |
| `access-required` | A book or paper this run did not read. |

Copy at most 25 words of a caption or a page into `quote`. Longer text is a paraphrase. Keep `video_id` plus timestamp, or the page URL. Do not paste a transcript into the final document.

## 7. Locked decisions

These are not suggestions.

1. Normal pass reads captions only. It does not call `search_web`. It does not open articles. It does not open books.
2. Deep pass runs once per topic, after the agenda exists. It runs on concepts, not on videos.
3. A concept needs two different videos before it enters the deep pass. Same channel and same video id do not count as two.
4. At most 12 concepts go to the deep pass. Rank by number of distinct videos, then by number of briefs that listed the concept. If there is a tie, keep the concept whose name is alphabetically first, so the choice is stable. Concepts past 12 are listed in the final document under “Not researched this run” and have no web research.
5. If zero concepts are `multi-source`, skip the web. The final document says so, includes the video briefs’ steps marked `single-source`, and does not invent a consensus.
6. Deep-pass concepts run in series. Do not use a thread pool, `asyncio.gather`, or parallel LangGraph branches for the deep pass. The normal-pass extraction may use a thread pool of size 2. Not more.
7. `max_videos` is an integer from 1 to 50. Default 10. Above 50, `POST /research` and `research_topic` return an error whose text is exactly `max_videos must be 1 to 50`. Change `ReportRequest` from `le=200` to `le=50`.
8. Do not download video files. Do not download book PDFs. Do not search `filetype:pdf`, LibGen, Anna’s Archive, Z-Library, or PDFDrive.
9. A book named by a speaker is a lead. If you cannot open three independent lawful summaries, mark that book `access-required` and continue the concept from public pages. Do not fill missing chapters from model memory.
10. The final document is organized by what a person does. It is not organized by video, by speaker, or by book.
11. Legacy mode stays available. `LEGACY_ATOM_PIPELINE=true` runs today’s `run_pipeline` unchanged. Default is the new pipeline.
12. `runs/` is gitignored. `plan4me.db` stays gitignored. `.env` stays gitignored.

## 8. Stages, in order

The job object’s `stage` field uses these names, in this order:

```text
search
transcribe
video_briefs
merge_agenda
deep_research
final_doc
```

`backend/pipeline/jobs.py` `STAGE_ORDER` becomes that tuple. `frontend/app/api.ts` `ResearchStage` becomes those strings plus `"done"`. `ResearchScene.tsx` `STAGE_INDEX` becomes search 0, transcribe 1, video_briefs 2, merge_agenda 3, deep_research 4, final_doc 5, done 6. Do not redesign the Three.js scene. If a count is missing, the scene may ignore it. The typecheck must pass.

`completed_stages` appends a name only after that stage’s files exist on disk.

### 8.1 search

Call the existing `search_videos(topic, settings.num_candidates)`. `num_candidates` stays 50 unless the environment overrides it. Do not raise it to match `max_videos`.

Write `runs/<job_id>/search.json`: a JSON list of `VideoMeta` objects, YouTube order preserved.

### 8.2 transcribe

Call the existing `select_and_transcribe(candidates, languages, target=max_videos)`.

Write `runs/<job_id>/transcripts/<video_id>.json` for each transcript. The JSON is the existing `Transcript` model: `video_id`, `title`, `channel`, `language`, `source`, `snippets` (`text`, `start`, `duration`). This file is the caption record. It is not pasted into the final document.

Write `runs/<job_id>/skipped.json`: a list of `{video_id, title, reason}`. `reason` is `no_captions` when both captions and Deepgram produced nothing.

If `transcripts_found` is 0, stop the job with status `error` and `error` text exactly `no captions found for this topic`. Do not call the models.

### 8.3 video_briefs (normal pass)

For each transcript, call `write_video_brief(topic, transcript) -> VideoBrief`.

Implement this in a new file `backend/research/video_brief.py`. Use `get_extraction_llm()` and `structured(...)` with the schema in section 9. If the transcript’s `full_text` is longer than 24000 characters, split it into chunks of at most 12000 characters on snippet boundaries, extract a partial brief per chunk, then call the model once more to merge those partial briefs into one `VideoBrief`. The merge prompt says: drop duplicates, keep every disagreement, keep every number, do not add a topic that was not in a partial brief.

The system prompt for extraction is this text, copied exactly:

```text
You extract a research brief from one video transcript. You only use the transcript. You do not add advice from your training. You do not search.

Return topics, claims, ordered processes, checklist items, named books, named people, named tools, disagreements inside this video, and numbers.

A topic is a named idea a person could act on or argue with: a rule, a stage, a distinction, a failure mode, or a question. A metaphor that is only a metaphor is not a topic.

Each claim has the seconds where the speaker says it. Copy at most 25 words into quote. Paraphrase the claim.

A number has a label. Use self-reported when the speaker describes their own results. Use estimate when they say the number is rough. Use heuristic when it is a rule of thumb. Never use verified.

If the speaker contradicts themselves, put both sides in disagreements. Do not pick a winner.
```

The human message is:

```text
TOPIC: <topic>
VIDEO_ID: <video_id>
TITLE: <title>
CHANNEL: <channel>

TRANSCRIPT:
[<start_seconds>] <text>
...
```

One line per snippet. `start_seconds` is the integer `snip.start`.

Write:

- `runs/<job_id>/videos/<video_id>.json` — the `VideoBrief`
- `runs/<job_id>/videos/<video_id>.md` — rendered from the brief with the headings in section 10

If the model returns no brief or a brief with zero topics and zero claims, write the JSON with an `error` field `empty brief` and continue with the other videos. Do not fail the whole job unless every video fails. If every video fails, status `error`, text `every video brief failed`.

A thread pool of size 2 may run `write_video_brief`. The merge in 8.4 starts only after every brief has been written or marked failed.

### 8.4 merge_agenda

Implement `build_agenda(topic, briefs: list[VideoBrief]) -> Agenda` in `backend/research/agenda.py`.

Input is the successful briefs only. The model sees the topic names, claim texts, process step texts, checklist texts, book titles, and `video_id`s. It does not see full transcripts.

System prompt, copied exactly:

```text
You merge video briefs into one research agenda. You do not add topics that no brief contains. You do not search.

Two topics are the same concept when a person following one would do the same action as a person following the other. Different names for that action are one concept. Use the shortest name that still says the action.

support_count is the number of distinct video_id values that contain the concept. Set label to multi-source when support_count is 2 or more. Set label to single-source when support_count is 1.

If two videos give different numbers or opposite actions for the same concept, set contested to true and write both sides. Do not average numbers.

books are titles a speaker named. Copy the video_id that named each title. A book is not a source you read.
```

After the model returns, your code, not the model, does this check:

- Drop any concept whose `video_ids` are not a subset of the briefs you passed in.
- Recompute `support_count` as `len(set(video_ids))`.
- Set `label` from that count, ignoring the model’s label if they disagree.
- Sort `multi-source` concepts by `support_count` descending, then by `name` ascending.
- Set `deep_research` true on the first 12. Set it false on the rest.
- `single-source` concepts have `deep_research` false.

Write `runs/<job_id>/agenda.json`.

### 8.5 deep_research

Implement `research_concept(topic, concept: AgendaConcept) -> ConceptNote` in `backend/research/deep.py`.

Call it in a for-loop over concepts where `deep_research` is true. Persist each note before starting the next concept, so a crash can resume. On resume, if `runs/<job_id>/concepts/<slug>.json` exists, skip that concept.

`<slug>` is the concept name lowercased, spaces turned into hyphens, characters other than `a-z`, `0-9`, and `-` removed, truncated to 60 characters. If two concepts would share a slug, append `-2`, then `-3`.

For each concept, do Pass A only when `concept.books` is non-empty. Then always do Pass B. Pass A does not replace Pass B.

#### Pass A — map a named book

For each book on that concept, at most 2 books (if the concept names more, keep the two that were named by the most videos, then alphabetical title):

Search queries, in this order, stop after you have 4 kept summaries or you have run all three queries:

1. `"<title>" "<author>" detailed summary` — omit `"<author>"` when the brief has no author
2. `"<title>" chapter summary`
3. `"<title>" book notes`

Do not run a query that contains `pdf`, `filetype:pdf`, `libgen`, `download`, or `free book`.

Open the hit. Keep it only if all of these are true:

- Extracted text is at least 800 words.
- The page names the book title.
- You can see a person or site name that wrote the summary.
- The page is not a paste of whole chapters. If more than 30 consecutive lines look like book prose with no summary voice, discard the page and set `rejected_reason` to `looks like a book paste`.
- It is not the same article as one you already kept. Same host and a title that shares the first 40 characters counts as the same article.

You need 3 kept summaries to map the book. 4 is the maximum you keep. If you have 0, 1, or 2, write the book as `access-required` and an empty concept list from summaries. Do not invent chapters.

From the kept summaries, list sub-concepts that appear in at least two summaries. Each sub-concept has `name`, `agreed_definition` (one or two sentences), `summary_urls`, `disagreement` or the exact string `no disagreement found`, and `operable_question`. Mark every sub-concept `summary-derived`.

These sub-concepts are context for Pass B. They are not yet steps in the final document. A step enters the final document only after Pass B opens a public source that describes that action.

#### Pass B — public sources on the concept

Run these queries. `<field>` is the user topic (the string they typed), not a guessed industry.

1. `<concept name> <field>`
2. `<concept name> talk` when a brief named a person for this concept, add that person’s name: `<concept name> <person> talk`
3. `<concept name> limitation`
4. `<concept name> critique`

Do not query `summary of <book>` in Pass B. That was Pass A.

For each query, take at most 5 hits. Across all queries, open at most 8 pages. Stop early if you already have 4 kept pages.

A kept page has all of these:

- Extracted text is at least 500 words.
- It describes steps, a failure, or a distinction. A page that only defines the term in one paragraph and then sells a course is discarded (`rejected_reason`: `too thin`).
- It is not a video you already transcribed in this job. If the URL is `youtube.com` or `youtu.be`, skip it. Captions were the normal pass. Do not fetch them again here.
- The host is not in the block list in section 11.

Record for every page you attempted, kept or not: `url`, `title`, `fetched_at` (ISO-8601 date), `word_count`, `kept` (true or false), `rejected_reason` (empty when kept).

Write the concept note with the headings in section 10. Each step in “What to do” includes the URL or, when the only support is the videos, the `video_id` and timestamp. Steps that only the book summaries mentioned stay under “Summary-derived, not yet operable” and do not appear in “What to do”.

If you open zero pages that pass the rules, the concept note’s “What to do” is empty, “Not established” says `no public page opened that passed the keep rules`, and the video claims remain available as `single-source` relative to the web (they may still be `multi-source` across videos). Do not fill “What to do” from model memory.

### 8.6 final_doc

Implement `write_final_document(topic, agenda, notes, briefs) -> str` in `backend/research/final_doc.py`. Use `get_synthesis_llm()`.

The model receives: the agenda JSON, each concept note’s Markdown, and from each video brief only the disagreements and the numbers. It does not receive full transcripts.

System prompt, copied exactly:

```text
You write one operating document from research notes. You do not add steps, numbers, books, or tools that are not in the notes or the agenda.

Organize by what a person does, in order. Do not organize by video or by book.

A required step must be multi-source in the notes. A single-source point goes in the appendix, not in the required steps.

Where notes disagree, keep both sides in Disagreements. Do not average numbers.

Every number you mention appears again in the Numbers table with its label.

If a section has nothing, write the sentence "Nothing in the sources opened." Do not delete the heading.
```

Required headings, in this order, level-2:

```text
## How this document was made
## What to do
## What not to do
## Disagreements
## Numbers
## Concepts researched
## Single-source appendix
## Books named but not read
## Not researched this run
## Not established
```

“How this document was made” is written by your code, not the model, and prepended after the model returns. It contains: topic, job id, date, candidates searched, transcripts used, video ids skipped, concepts that received a deep pass, and the sentence `Video figures are self-reported or heuristic unless a row in Numbers says otherwise.`

If the model omits a heading, your code inserts that heading and the sentence `Nothing in the sources opened.`

Write `runs/<job_id>/FINAL.md`. Also save it as the `markdown` field of the `KnowledgeReport` so the existing UI reader still shows it.

## 9. Schemas

Add these to `backend/schemas.py`. Use Pydantic. Fields are required unless marked optional.

### VideoBrief

| Field | Type | Rule |
| --- | --- | --- |
| `video_id` | str | From the transcript, not the model |
| `title` | str | From the transcript |
| `channel` | str or null | From the transcript |
| `topics` | list of `{name, what_it_is}` | `name` is the short concept name. `what_it_is` is one sentence |
| `claims` | list of `{text, timestamp_seconds, quote, label}` | `label` is one of `self-reported`, `estimate`, `heuristic`, `single-source`. Default `single-source` when the model omits it. `timestamp_seconds` is a number |
| `processes` | list of `{name, steps}` | `steps` is a list of `{text, timestamp_seconds}` in the order the speaker gave |
| `checklist` | list of `{text, timestamp_seconds}` | |
| `books` | list of `{title, author}` | `author` may be null. Only books the speaker named |
| `people` | list of str | People the speaker named, other than themselves |
| `tools` | list of str | |
| `disagreements` | list of `{side_a, side_b}` | Empty list when the speaker does not contradict themselves |
| `numbers` | list of `{figure, label, timestamp_seconds, context}` | `figure` is the number as a string, so `10k` stays `10k`. `label` is `self-reported`, `estimate`, or `heuristic` |

Your code overwrites `video_id`, `title`, and `channel` with the transcript values after the model returns.

### Agenda

| Field | Type |
| --- | --- |
| `topic` | str |
| `concepts` | list of AgendaConcept |

### AgendaConcept

| Field | Type | Rule |
| --- | --- | --- |
| `name` | str | |
| `what_videos_agree` | str | One or two sentences. Empty string if they do not agree |
| `video_ids` | list of str | |
| `support_count` | int | Recomputed by code |
| `label` | `multi-source` or `single-source` | Recomputed by code |
| `contested` | bool | |
| `contest` | str | What the two sides are. Empty when `contested` is false |
| `operable_question` | str | What would a person do differently if this is true? |
| `books` | list of `{title, author, video_id}` | |
| `deep_research` | bool | Set by code, not the model |

### ConceptNote

Stored as JSON plus the Markdown file.

| Field | Type |
| --- | --- |
| `concept` | str |
| `slug` | str |
| `book_maps` | list of BookMap |
| `pages` | list of PageRecord |
| `markdown` | str |

### BookMap

| Field | Type |
| --- | --- |
| `title` | str |
| `author` | str or null |
| `access` | `access-required` or `summaries-mapped` |
| `summaries` | list of `{url, author, word_count, fetched_at}` |
| `subconcepts` | list of `{name, agreed_definition, summary_urls, disagreement, operable_question, label}` |

`label` on a subconcept is always `summary-derived`. `access` is `summaries-mapped` only when `len(summaries) >= 3`. Otherwise `access-required` and `subconcepts` is empty.

### PageRecord

| Field | Type |
| --- | --- |
| `url` | str |
| `title` | str |
| `fetched_at` | str |
| `word_count` | int |
| `kept` | bool |
| `rejected_reason` | str |

### KnowledgeReport counts

Add these fields. Keep the old fields so the UI does not crash. Set them as follows. Do not leave them as the old meanings.

| Field | New meaning |
| --- | --- |
| `videos_processed` | Number of transcripts used. Not the candidate count |
| `transcripts_found` | Same number as `videos_processed` |
| `atoms_extracted` | Number of claims across all video briefs |
| `clusters` | Number of agenda concepts |
| `candidates_searched` | New field. Length of the search list |
| `videos_skipped` | New field. Length of `skipped.json` |
| `concepts_deep_researched` | New field. How many concept notes were written |

`insights` may stay an empty list for the new pipeline. Do not fake cluster cards.

## 10. Markdown the files must use

### Video brief `videos/<video_id>.md`

```text
# <title>

Video: https://www.youtube.com/watch?v=<video_id>
Channel: <channel>
This file is a caption brief. No web page was opened for it.

## Topics
## Claims
## Processes
## Checklist
## Books named
## Numbers
## Disagreements inside this video
```

Under Claims, each bullet ends with `(<video_id> <timestamp_seconds>s, <label>)`.

### Concept note `concepts/<slug>.md`

```text
# <concept name>

User topic: <topic>
This note: public sources opened on <YYYY-MM-DD>.
Book map: <access-required or summaries-mapped>. The book was not read, unless access is a licensed file, which this pipeline does not have.

## What to do
## What not to do
## Disagreement
## Numbers
## Summary-derived, not yet operable
## Pages opened
## Not established
```

“What to do” is a numbered list. Each step ends with a locator in parentheses: a URL, or `video <id> at <seconds>s`.

“Numbers” is a table with columns `figure`, `label`, `who`, `where`.

If “Disagreement” has no content, the body is the exact sentence `no disagreement found in the sources opened.`

### Final document

Headings are fixed in section 8.6. “Numbers” is a Markdown table with columns `figure`, `label`, `who`, `where`. “Books named but not read” lists title, author if known, and `access-required` or `summaries-mapped`. “Single-source appendix” lists the concept name and the one `video_id`. Those items are not written as instructions.

## 11. Web search and page fetch

New file `backend/research/web.py`.

```text
search_web(query: str, max_results: int) -> list[WebHit]
fetch_page(url: str) -> FetchedPage
```

`WebHit`: `title`, `url`, `snippet`.

`FetchedPage`: `url`, `title`, `text`, `word_count`, `error`. `error` is null on success.

Provider, environment variable `WEB_SEARCH_PROVIDER`:

| Value | Behavior |
| --- | --- |
| `duckduckgo` | Default. Use the `ddgs` package (`DDGS().text(query, max_results=...)`). No API key. |
| `tavily` | Used only when `WEB_SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY` is non-empty. POST `https://api.tavily.com/search` with `search_depth` `basic`, `max_results` as passed. If the key is empty, raise a startup error for that request whose text is `TAVILY_API_KEY is empty`. Do not silently fall back. |

Add `ddgs` and `trafilatura` and `httpx` to `requirements.txt` if they are not already pinned. Fetch with `httpx.Client(timeout=20.0, follow_redirects=True)`. Extract main text with `trafilatura.extract`. User-Agent string: `plan4me-research`.

Do not fetch a URL whose host contains any of these substrings:

```text
libgen
annas-archive
z-lib
zlib
pdfdrive
sci-hub
```

If the URL path ends in `.pdf`, do not download it. Record `rejected_reason` `pdf not fetched`.

Word count is `len(text.split())`.

Cache fetches in `runs/<job_id>/pages/<sha1 of url>.json` so a retry does not fetch twice. The JSON is the `FetchedPage` plus `fetched_at`.

## 12. Where the code goes

Create these files. Do not put this logic inside `extraction.py` or `synthesis.py`. Those stay on the legacy path.

```text
backend/research/__init__.py
backend/research/video_brief.py
backend/research/agenda.py
backend/research/web.py
backend/research/deep.py
backend/research/final_doc.py
backend/research/pipeline.py
backend/research/render.py
```

`pipeline.py` exposes `run_research(topic, max_videos, languages, job_id) -> KnowledgeReport` and `stream_research(...)` yielding `(stage_name, counts_dict)` after each stage, so `jobs.py` can keep its loop.

`graph.py` `run_pipeline` stays the legacy function. `jobs.py` `_run` calls `stream_research` unless `get_settings().legacy_atom_pipeline` is true, in which case it calls today’s `stream_pipeline`.

Add to `Settings`:

```text
legacy_atom_pipeline: bool = False
web_search_provider: str = "duckduckgo"
tavily_api_key: str = ""
deep_max_concepts: int = 12
```

`deep_max_concepts` default 12, and the code clamps it to the range 1–12 even if the environment sets 50. The cap of 12 is a product decision, not a setting the UI exposes.

Wire `mcp_server/server.py` `research_topic` to `run_research`. When `legacy_atom_pipeline` is true, it keeps calling `run_pipeline`.

`POST /report` calls the same function as `POST /research`, so the two endpoints cannot drift. `/report` still blocks until the job finishes. Set a note in the docstring that a deep run can take longer than the old caption run. Do not add a timeout that kills it at 60 seconds.

SQLite: extend `save_report` so the new count fields survive `GET /reports/latest`. You may store the extra counts inside the existing `meta` JSON. Do not create a second database file.

Disk layout:

```text
runs/<job_id>/search.json
runs/<job_id>/skipped.json
runs/<job_id>/transcripts/<video_id>.json
runs/<job_id>/videos/<video_id>.json
runs/<job_id>/videos/<video_id>.md
runs/<job_id>/agenda.json
runs/<job_id>/concepts/<slug>.json
runs/<job_id>/concepts/<slug>.md
runs/<job_id>/pages/<sha1>.json
runs/<job_id>/FINAL.md
```

Add `runs/` to `.gitignore`.

`job_id` for `POST /report` and for MCP `research_topic` is a new 12 hex character id, the same generator `jobs.py` already uses. Return it on the report as `job_id`. Add the field to `KnowledgeReport`.

Resume: `run_research` accepts an optional `resume_job_id`. When that folder exists, it loads files that are already there and starts at the first missing stage. MCP does not need a new tool for resume in this assignment. `GET /research/{job_id}` is enough for the UI. If you add a resume path, it is a function argument used by tests, not a new public concept.

## 13. UI

The page already polls `GET /research/{job_id}` and renders `report.markdown`. Keep that. Update the stage type so the progress label can show the six names. Human labels, in order:

```text
Searching YouTube
Reading captions
Writing a brief for each video
Combining topics
Researching the combined topics
Writing the final document
```

Do not add a second results view. The Markdown reader is the product.

## 14. Tests

Put tests in `tests/test_research_pipeline.py`. They must pass with the network disabled. Monkeypatch `search_web` and `fetch_page` and the LLM.

Required cases:

1. **Merge support.** Two fixture briefs both contain a topic whose action is “do not publish the unbuilt roadmap”. Different wording is fine. `build_agenda` result contains one concept with `support_count` 2, `label` `multi-source`, `deep_research` true. The test may call the pure Python recomputation if you split `apply_agenda_rules(raw, briefs) -> Agenda` out of the model call. Test `apply_agenda_rules`, not the live model.
2. **Single video topic.** A topic present in one brief only has `label` `single-source` and `deep_research` false.
3. **Cap 12.** 15 multi-source concepts go in. Exactly 12 have `deep_research` true. Those 12 are the ones with the highest `support_count`. The other 3 appear in the agenda with `deep_research` false.
4. **Model cannot inflate support.** The raw model output says `support_count` 9 and lists one `video_id`. After `apply_agenda_rules`, `support_count` is 1 and `label` is `single-source`.
5. **Unknown video id dropped.** A concept that cites a `video_id` not in the briefs is removed.
6. **Blocked host.** `fetch_page` is not called for `https://libgen.example/book`. `search_web` results containing that host are dropped before fetch. Test the filter function `url_allowed(url) -> bool`.
7. **PDF.** `url_allowed` returns false for a URL ending in `.pdf`.
8. **Book map threshold.** A helper `book_access(summaries) -> str` returns `access-required` for 2 summaries and `summaries-mapped` for 3.
9. **Empty captions.** `select` result empty stops the pipeline with `no captions found for this topic` and does not call the brief model. Test with a fake `select_and_transcribe` that returns `[]`.
10. **Final headings.** Given a stub model that returns Markdown missing `## Numbers`, `ensure_headings(markdown)` inserts `## Numbers` and `Nothing in the sources opened.`
11. **max_videos.** A request with `max_videos=51` is rejected with `max_videos must be 1 to 50`.

Live check, not part of `pytest` by default. Mark it `@pytest.mark.live` and exclude that mark in the default `pytest` addopts if you add addopts. Run it once by hand:

```powershell
Set-Location "C:\Users\affan\Affan Projects\plan4me"
.\.venv\Scripts\python.exe -m pytest tests/test_research_pipeline.py -q -m live
```

The live test uses topic `how to build in public`, `max_videos=3`, languages `en`. It asserts `transcripts_found >= 1`, `FINAL.md` exists, every `##` heading in section 8.6 is present, and `concepts/` has a note only for concepts with two or more of those three videos. If the three videos share no concept, `concepts/` is empty and `FINAL.md` says that no concept was multi-source. That outcome passes. A final document that instructs the reader to do something no brief and no page stated does not pass: you check this by reading `FINAL.md` yourself and confirming each required step’s locator exists in a concept note or a brief.

Paste the `runs/<job_id>/` path into the pull request.

## 15. Order of work

Do the phases in order. Do not start phase 5 before phase 2’s tests pass.

1. Schemas, `url_allowed`, `apply_agenda_rules`, `ensure_headings`, `book_access`. Tests 1–8 and 10–11. No YouTube. No model.
2. `render.py` so a fixture `VideoBrief` writes the Markdown in section 10. Read the file in the test and check the headings.
3. `video_brief.py` with a fake model that returns a fixed `VideoBrief`. One transcript in, one JSON and one Markdown out.
4. `web.py` with DuckDuckGo behind an interface. Unit tests use a fake. Do not hit the network in unit tests.
5. `deep.py` with fake search and fake pages. Assert Pass B does not run a query containing the book title plus `summary of`. Assert a YouTube URL is not fetched. Assert a concept with zero kept pages has an empty “What to do”.
6. `final_doc.py` with a fake synthesis model.
7. `pipeline.py` wiring, `jobs.py` stage names, API field `job_id`, MCP, frontend stage union.
8. `.gitignore`, `.env.example` rows for `LEGACY_ATOM_PIPELINE`, `WEB_SEARCH_PROVIDER`, `TAVILY_API_KEY`, `DEEP_MAX_CONCEPTS`.
9. Legacy flag still calls `stream_pipeline`. One test: with the flag true, `jobs._run`’s pipeline function is the old one. Monkeypatch to assert which function was entered. Do not call Bedrock in that test.
10. Live check with 3 videos. Then stop.

## 16. What you must not build

- A worker that web-researches each video before the agenda exists.
- A second synthesis that averages conflicting numbers.
- Cosine clustering as the way concepts are chosen. Leave it on the legacy path only.
- A PDF downloader, a book scanner, or a “full text of the book” prompt.
- A new database product. SQLite plus `runs/` is the store.
- A new frontend framework, auth, or accounts.
- Changes to the extraction prompt of the legacy atom pipeline, unless a legacy test you added requires a bugfix. Do not “improve” that prompt in this assignment.
- Parallel deep research.
- More than 12 deep concepts.
- Fetching YouTube again during the deep pass.
- Printing `.env`, AWS keys, or `TAVILY_API_KEY`.

## 17. How you know a sentence is allowed in the final document

Before you leave a sentence in `## What to do` or `## What not to do`, point at one of these:

- Two video ids in the agenda for that concept, and the concept note repeats the action, or
- One kept page URL in that concept’s `pages` list, and a second independent source (another kept page, or a second video) says the same action.

If you cannot point at those, the sentence moves to `## Single-source appendix` or `## Not established`.

A book summary can justify a sentence in `## Summary-derived, not yet operable` inside the concept note. It cannot justify a required step.

## 18. Fixture you can copy into tests

Use these two briefs as the merge fixture. They are intentionally short.

Brief A, `video_id` `aaa`:

- Topic name: `Do not share the unbuilt roadmap`. What it is: do not announce features that are not shipped.
- Claim: “Sharing the roadmap invites copies.” timestamp 120. label `heuristic`.
- Book: none.

Brief B, `video_id` `bbb`:

- Topic name: `Keep unshipped features private`. What it is: do not post the feature plan before it exists.
- Claim: “We lost a deal after posting the roadmap.” timestamp 40. label `self-reported`.
- Book: `{title: "Obviously Awesome", author: "April Dunford"}`.

Expected agenda after `apply_agenda_rules`: one `multi-source` concept, `support_count` 2, `contested` false, `books` length 1, `deep_research` true. The concept name can be either of the two names or a shorter name the model chose; the test of the pure function should force the name by passing an already merged raw concept so you are not testing the model’s taste. The model-taste test is the live check, read by you.

Add a third brief `ccc` whose only topic is `Post a daily screenshot`. Expected: that concept is `single-source`, `deep_research` false.

## 19. Pull request

Branch name: `research-pipeline`.

Commit message:

```text
Add a two-depth research pipeline.

Video briefs stay on captions. One deep pass researches only the concepts that show up in two or more videos, then writes a single operating document.
```

Pull request body includes:

- The live `runs/<job_id>/` path
- How many transcripts, how many multi-source concepts, how many pages kept
- Confirmation that `LEGACY_ATOM_PIPELINE=true` still enters `stream_pipeline`
- Confirmation that no PDF was fetched

Do not commit `runs/`, `plan4me.db`, or `.env`.
