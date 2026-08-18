"""
Resume Screening Agent
-----------------------
Takes a Job Description + a folder of resumes, ranks the resumes by
relevance using TF-IDF cosine similarity, and (optionally) asks an LLM
to explain each candidate's ranking in plain English.

Usage:
    python agent.py --jd data/job_description.txt --resumes data/resumes --out output/ranked_results.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: only imported/used if an API key is present and --explain is passed
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------

def extract_text(path: Path) -> str:
    """Extract raw text from .txt, .pdf, or .docx files."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            sys.exit("pypdf is required for PDF resumes. Run: pip install pypdf")
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix == ".docx":
        try:
            import docx
        except ImportError:
            sys.exit("python-docx is required for DOCX resumes. Run: pip install python-docx")
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {path.name}")


def clean_text(text: str) -> str:
    """Light normalization: collapse whitespace, lowercase for matching."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Lightweight skill/keyword extraction (heuristic, no ML training needed)
# ---------------------------------------------------------------------------

# A small, extensible skill vocabulary. In a real product this would be a
# much larger taxonomy or an LLM call; kept simple here to stay transparent
# and debuggable within a 24h build.
SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "react", "node.js", "node",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "aws", "azure", "gcp",
    "docker", "kubernetes", "git", "html", "css", "django", "flask",
    "fastapi", "machine learning", "deep learning", "nlp", "pandas", "numpy",
    "tensorflow", "pytorch", "rest api", "graphql", "ci/cd", "agile", "scrum",
    "excel", "tableau", "power bi", "linux", "c++", "c#", ".net", "spring",
]

EDU_KEYWORDS = ["b.e", "b.tech", "m.tech", "bachelor", "master", "phd", "diploma", "engineering"]


def extract_skills(text: str) -> list:
    lowered = text.lower()
    return sorted({s for s in SKILL_VOCAB if s in lowered})


def extract_education(text: str) -> list:
    lowered = text.lower()
    return sorted({e for e in EDU_KEYWORDS if e in lowered})


def extract_experience_years(text: str) -> str:
    """Heuristic: look for patterns like '3 years', '5+ years of experience'."""
    match = re.search(r"(\d+)\+?\s*years?", text.lower())
    return match.group(1) if match else "unknown"


# ---------------------------------------------------------------------------
# 3. Scoring: TF-IDF + cosine similarity against the Job Description
# ---------------------------------------------------------------------------

def score_resumes(jd_text: str, resume_texts: dict) -> dict:
    """
    Returns {filename: similarity_score} where similarity_score is 0-1,
    computed via TF-IDF cosine similarity between each resume and the JD.
    This is deterministic and explainable -- no LLM call needed to get a
    ranked list, which avoids hallucinated scores.
    """
    filenames = list(resume_texts.keys())
    corpus = [jd_text] + [resume_texts[f] for f in filenames]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]

    return {filenames[i]: round(float(similarities[i]), 4) for i in range(len(filenames))}


# ---------------------------------------------------------------------------
# 4. Optional: LLM-generated reasoning for each candidate
# ---------------------------------------------------------------------------

def generate_reasoning(jd_text: str, resume_text: str, skills: list, score: float) -> str:
    """
    Calls the Claude API to produce a short, human-readable explanation of
    why this candidate scored the way they did. Falls back to a templated
    explanation if no API key is configured.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key or Anthropic is None:
        # Deterministic fallback -- keeps the agent runnable with zero API cost.
        if skills:
            return f"Matched skills: {', '.join(skills[:6])}. Similarity score: {score}."
        return f"No strong keyword overlap detected with the job description. Similarity score: {score}."

    client = Anthropic(api_key=api_key)
    prompt = (
        "You are an HR assistant. In 2-3 concise sentences, explain why this "
        "resume is or isn't a good match for the job description below. "
        "Be specific about matched or missing skills. Do not repeat the full "
        "resume or JD text back.\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:2000]}\n\n"
        f"RESUME:\n{resume_text[:2000]}\n\n"
        f"Computed similarity score (0-1): {score}"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"(LLM reasoning unavailable: {e}). Similarity score: {score}."


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------

def run(jd_path: str, resumes_dir: str, out_path: str, explain: bool):
    jd_raw = extract_text(Path(jd_path))
    jd_text = clean_text(jd_raw)

    resumes_dir = Path(resumes_dir)
    resume_files = [
        p for p in sorted(resumes_dir.iterdir())
        if p.suffix.lower() in (".txt", ".pdf", ".docx")
    ]

    if not resume_files:
        sys.exit(f"No resumes found in {resumes_dir} (expected .txt, .pdf, or .docx)")

    resume_texts_raw = {p.name: extract_text(p) for p in resume_files}
    resume_texts_clean = {name: clean_text(t) for name, t in resume_texts_raw.items()}

    scores = score_resumes(jd_text, resume_texts_clean)

    results = []
    for name in resume_texts_raw:
        raw = resume_texts_raw[name]
        skills = extract_skills(raw)
        education = extract_education(raw)
        experience = extract_experience_years(raw)
        score = scores[name]

        reasoning = ""
        if explain:
            reasoning = generate_reasoning(jd_text, resume_texts_clean[name], skills, score)

        results.append({
            "filename": name,
            "score": score,
            "skills_matched": ", ".join(skills),
            "education": ", ".join(education),
            "years_experience_mentioned": experience,
            "reasoning": reasoning,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    write_csv(results, out_path)
    write_json(results, out_path.replace(".csv", ".json"))

    print(f"\nRanked {len(results)} resumes against {jd_path}\n")
    for r in results:
        print(f"  #{r['rank']:<2} {r['filename']:<30} score={r['score']}")
    print(f"\nFull results written to:\n  {out_path}\n  {out_path.replace('.csv', '.json')}")


def write_csv(results: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["rank", "filename", "score", "skills_matched", "education",
                  "years_experience_mentioned", "reasoning"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_json(results: list, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# 6. CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", default="data/job_description.txt", help="Path to job description text file")
    parser.add_argument("--resumes", default="data/resumes", help="Folder containing resume files")
    parser.add_argument("--out", default="output/ranked_results.csv", help="Output CSV path")
    parser.add_argument("--explain", action="store_true",
                         help="Use Claude API to generate per-candidate reasoning (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    run(args.jd, args.resumes, args.out, args.explain)
