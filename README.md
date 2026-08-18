# Resume Screening Agent

Ranks a folder of resumes against a job description and outputs a scored,
ordered shortlist with matched skills and (optionally) an LLM-written
explanation for each candidate's rank.

**My agent takes a job description + a folder of resumes and produces a
ranked CSV/JSON shortlist with scores and reasoning.**

---

## How it works

1. **Extract text** from each resume (`.txt`, `.pdf`, `.docx`) and the job
   description.
2. **Score relevance** using TF-IDF + cosine similarity between the JD and
   each resume. This is deterministic — the same input always gives the
   same score, and the score is fully explainable by which words/phrases
   overlap. No LLM call is required to produce the ranking, which avoids
   the ranking itself being a hallucination.
3. **Extract structured fields** (skills, education level, mentioned years
   of experience) using a keyword vocabulary — again deterministic, not
   LLM-generated, so it's fast, free, and auditable.
4. **(Optional) Generate reasoning** — if you pass `--explain` and set an
   `ANTHROPIC_API_KEY`, the agent calls Claude once per resume to write a
   short human-readable explanation of the fit. Without a key, it falls
   back to a templated explanation built from the matched skills, so the
   agent is always runnable even with zero API cost.
5. **Rank and output** a CSV and JSON file, sorted by score, highest first.

```
resumes + JD  →  extract text  →  TF-IDF similarity  →  rank
                                          ↓
                              skill/education extraction
                                          ↓
                          (optional) LLM reasoning per resume
                                          ↓
                                 ranked_results.csv/json
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd resume-screening-agent
pip install -r requirements.txt
```

Requires Python 3.9+.

### 2. (Optional) Configure your API key

Only needed if you want LLM-generated reasoning (`--explain`). Without it,
the agent still runs and produces scores + matched skills, just with a
templated explanation instead of a natural-language one.

```bash
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=your_key_here   # or use a tool like python-dotenv / direnv
```

### 3. Run it

```bash
python agent.py
```

This uses the bundled sample data (`data/job_description.txt` and
`data/resumes/`) and writes results to `output/ranked_results.csv` and
`output/ranked_results.json`.

To use your own data:

```bash
python agent.py --jd path/to/job_description.txt --resumes path/to/resumes_folder --out output/ranked_results.csv
```

To include LLM-generated reasoning per candidate:

```bash
python agent.py --explain
```

---

## Sample input/output

**Job description:** `data/job_description.txt` — a Junior Backend
Developer role requiring Python, SQL, REST APIs, and basic Docker/Git.

**Resumes:** `data/resumes/` — 5 sample resumes with deliberately varied
fit (two strong backend-dev matches, one Java/Spring developer, one data
scientist, one mechanical engineer) so the ranking is easy to sanity-check.

**Output** (`output/ranked_results.csv`, generated without `--explain`):

| rank | filename | score | skills_matched |
|------|----------|-------|-----------------|
| 1 | resume_priya.txt | 0.1729 | agile, aws, docker, fastapi, flask, git, postgresql, python, rest api, sql |
| 2 | resume_ananya.txt | 0.1617 | django, docker, git, linux, mysql, python, rest api, sql |
| 3 | resume_arjun.txt | 0.0670 | css, git, html, java, mysql, rest api, spring, sql |
| 4 | resume_karthik.txt | 0.0580 | excel, python |
| 5 | resume_fatima.txt | 0.0345 | machine learning, numpy, pandas, python, sql, tableau, tensorflow |

The two candidates with direct Python/REST API/SQL backend experience
rank highest; the mechanical engineer and data scientist — who have
some keyword overlap but aren't a role fit — rank lowest. This matches
what a human reviewer would conclude by eye, which is the sanity check
used throughout development.

Full JSON output (with `reasoning` field) is in `output/ranked_results.json`.

---

## Design choices & tradeoffs

- **TF-IDF over embeddings**: chosen for the core score because it's free,
  deterministic, and fully explainable without any API dependency — the
  agent runs and ranks correctly with zero cost or network access. The
  tradeoff is that it's a bag-of-words method: it won't catch semantic
  matches where a resume says "used Postgres" and the JD says "relational
  database" without any shared vocabulary at all (though with
  `ngram_range=(1,2)` it does catch adjacent-phrase overlap like "rest api").
  With more time, I'd add an embeddings-based score (e.g. `sentence-transformers`
  or Claude/OpenAI embeddings) and blend it with TF-IDF for a hybrid score.
- **Keyword vocabulary for skill extraction** rather than an LLM call: keeps
  the agent fast and cheap to run on a batch of resumes, and the extracted
  skills are directly auditable (you can see exactly why a skill was or
  wasn't detected). The tradeoff is the vocabulary is a fixed list — with
  more time I'd expand it or let the LLM extract a more open-ended skill
  set for resumes that use different terminology.
- **LLM reasoning is opt-in (`--explain`)**: the ranking itself never
  depends on the LLM, so reviewers can run the agent with no API key at
  all and still get a correct, ranked, explainable output. The LLM is
  used only to phrase an existing, already-computed result in natural
  language — it's not asked to invent a score.
- **Years-of-experience extraction is a simple regex** (`"\d+\+?\s*years?"`).
  It works for straightforward phrasing like "2 years" but would miss
  date-range calculations (e.g. inferring 3 years from "2021–2024"). With
  more time I'd add date-range parsing.
- **File formats**: `.txt`, `.pdf`, and `.docx` are all supported. PDF
  extraction quality depends on whether the source PDF has selectable
  text (scanned/image-only PDFs would need OCR, which is out of scope
  for this build).
- **Batch size**: tested with 5 resumes here to keep the sample readable,
  but the scoring approach (TF-IDF matrix over the whole corpus at once)
  scales to 10+ resumes with no code changes — it's already a single
  vectorized operation, not a per-resume loop.

## What I'd improve with more time

- Hybrid TF-IDF + embedding similarity score
- A richer, LLM-assisted skill taxonomy instead of a fixed keyword list
- Proper date-range parsing for experience duration
- OCR fallback for scanned/image PDF resumes
- A minimal web UI (drag-and-drop resumes + JD, view ranked table)
