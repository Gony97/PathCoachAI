#!/usr/bin/env python3
"""
Generate a top-15 learning modules catalog for data-science job search.

Pipeline:
  1) Collect job descriptions from public sources (Greenhouse, Lever, Ashby).
     (LinkedIn is intentionally NOT automated here due to ToS; see notes below.)
  2) Extract skills from descriptions using spaCy + PhraseMatcher with a curated taxonomy.
  3) Rank skills by frequency & signal and map them into 15 module definitions.
  4) Save outputs to modules_catalog.json and modules_catalog.md.

Usage:
  python job_skills_to_modules.py \
      --queries "Data Scientist" "Machine Learning Engineer" "Algorithms Engineer" \
      --greenhouse-orgs "openai" "airbnb" \
      --lever-orgs "stripe" \
      --ashby-orgs "anthropic" \
      --limit-per-source 50

Optional inputs:
  --profile-text path/to/your_profile.txt  # paste/export your LinkedIn profile to text and pass here

Notes about LinkedIn:
  Automating LinkedIn browsing can violate their Terms of Service and risk account restrictions.
  If you want to include LinkedIn signals, export your profile (or copy/paste to a text file)
  and supply via --profile-text. This script will incorporate it safely.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Tuple, Optional

# -----------------------------
# Networking utils (simple)
# -----------------------------
import time
import urllib.parse
import urllib.request


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def http_get(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


# -----------------------------
# Public job board collectors
# -----------------------------

class GreenhouseCollector:
    """Collects jobs from public Greenhouse boards for given orgs.
    API pattern: https://api.greenhouse.io/v1/boards/{org}/jobs
    """

    def __init__(self, orgs: List[str], limit: int = 50):
        self.orgs = orgs
        self.limit = limit

    def fetch(self, queries: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for org in self.orgs:
            try:
                url = f"https://api.greenhouse.io/v1/boards/{org}/jobs"
                data = json.loads(http_get(url))
                for job in data.get("jobs", [])[: self.limit]:
                    title = job.get("title", "")
                    if not self._title_matches(title, queries):
                        continue
                    jd = self._fetch_job_detail(job.get("absolute_url", ""))
                    if jd:
                        results.append({
                            "source": "greenhouse",
                            "org": org,
                            "title": title,
                            "location": job.get("location", {}).get("name"),
                            "absolute_url": job.get("absolute_url"),
                            "description_html": jd,
                        })
            except Exception as e:
                print(f"[Greenhouse:{org}] warn: {e}")
        return results

    def _fetch_job_detail(self, job_url: str) -> Optional[str]:
        if not job_url:
            return None
        try:
            html = http_get(job_url)
            # crude extraction of the JD from the page body
            m = re.search(r"<section[^>]*class=\"content\"[^>]*>([\s\S]+?)</section>", html, re.I)
            return m.group(1) if m else html
        except Exception:
            return None

    @staticmethod
    def _title_matches(title: str, queries: List[str]) -> bool:
        t = title.lower()
        return any(q.lower() in t for q in queries)


class LeverCollector:
    """Collects jobs from public Lever boards (JSON endpoints).
    Pattern: https://api.lever.co/v0/postings/{org}?mode=json
    """

    def __init__(self, orgs: List[str], limit: int = 50):
        self.orgs = orgs
        self.limit = limit

    def fetch(self, queries: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for org in self.orgs:
            try:
                url = f"https://api.lever.co/v0/postings/{org}?mode=json"
                data = json.loads(http_get(url))
                for job in data[: self.limit]:
                    title = job.get("text", "")
                    if not self._title_matches(title, queries):
                        continue
                    results.append({
                        "source": "lever",
                        "org": org,
                        "title": title,
                        "location": job.get("categories", {}).get("location"),
                        "absolute_url": job.get("hostedUrl"),
                        "description_html": job.get("description", ""),
                    })
            except Exception as e:
                print(f"[Lever:{org}] warn: {e}")
        return results

    @staticmethod
    def _title_matches(title: str, queries: List[str]) -> bool:
        t = title.lower()
        return any(q.lower() in t for q in queries)


class AshbyCollector:
    """Collects jobs from public Ashby boards (unauth JSON + HTML).
    We'll pull the board page and do a simple parse.
    """

    def __init__(self, orgs: List[str], limit: int = 50):
        self.orgs = orgs
        self.limit = limit

    def fetch(self, queries: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for org in self.orgs:
            try:
                board_url = f"https://jobs.ashbyhq.com/{urllib.parse.quote(org)}"
                html = http_get(board_url)
                # Find job links
                links = re.findall(r"href=\"(/[^\"]+/jobs/[^\"]+)\"", html)
                seen = set()
                for rel in links:
                    if len(results) >= self.limit:
                        break
                    if rel in seen:
                        continue
                    seen.add(rel)
                    job_url = urllib.parse.urljoin(board_url, rel)
                    try:
                        jhtml = http_get(job_url)
                        title_m = re.search(r"<title>([^<]+)</title>", jhtml)
                        title = title_m.group(1) if title_m else ""
                        if not self._title_matches(title, queries):
                            continue
                        # crude body grab
                        body_m = re.search(r"<article[\s\S]+?</article>", jhtml)
                        desc = body_m.group(0) if body_m else jhtml
                        results.append({
                            "source": "ashby",
                            "org": org,
                            "title": title,
                            "location": None,
                            "absolute_url": job_url,
                            "description_html": desc,
                        })
                    except Exception:
                        continue
            except Exception as e:
                print(f"[Ashby:{org}] warn: {e}")
        return results

    @staticmethod
    def _title_matches(title: str, queries: List[str]) -> bool:
        t = title.lower()
        return any(q.lower() in t for q in queries)


# -----------------------------
# Text processing / skill extraction
# -----------------------------

# We avoid heavy ML to keep dependencies light. spaCy small model is enough.
# The taxonomy below covers common DS/ML skills; you can extend via --extra-skill.

CURATED_SKILLS = [
    # Core
    "python", "sql", "git", "linux", "bash", "docker", "kubernetes", "airflow", "prefect",
    # Py data stack
    "pandas", "numpy", "scikit-learn", "sklearn", "matplotlib", "plotly", "seaborn",
    # ML/DL
    "tensorflow", "pytorch", "xgboost", "lightgbm", "catboost", "hugging face", "transformers",
    "mlflow", "weights & biases", "wandb",
    # Big data / cloud data
    "spark", "databricks", "hadoop", "hive", "snowflake", "bigquery", "redshift",
    # Cloud
    "aws", "gcp", "azure", "s3", "lambda", "ecs", "eks", "sagemaker",
    # Databases / warehouses
    "postgresql", "mysql", "mongodb", "clickhouse",
    # MLOps / orchestration
    "terraform", "ansible", "helm", "dbt",
    # Analytics / experimentation
    "statistics", "probability", "a/b testing", "causal inference", "experiment design",
    # NLP / CV / TS
    "nlp", "computer vision", "time series", "opencv", "spacy",
    # Viz / BI
    "tableau", "power bi", "looker",
    # Other
    "apis", "rest", "grpc"
]


def clean_html_to_text(html: str) -> str:
    # Remove tags and decode entities lightly
    text = re.sub(r"<(/?)(script|style)[^>]*>.*?</\1\2>", " ", html, flags=re.I|re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> List[str]:
    # simple tokens
    return re.findall(r"[a-zA-Z0-9+#/.&-]+", text.lower())


def extract_skills(texts: Iterable[str], extra_skills: List[str]) -> Dict[str, int]:
    skills = {s.lower(): 0 for s in CURATED_SKILLS + extra_skills}
    for t in texts:
        tokens = tokenize(t)
        joined = " ".join(tokens)
        for s in skills.keys():
            # match as whole word or phrase occurrences
            count = len(re.findall(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", joined))
            if count:
                skills[s] += count
    # prune zeroes
    return {k: v for k, v in skills.items() if v > 0}


# -----------------------------
# Module generation
# -----------------------------

@dataclass
class Module:
    id: str
    title: str
    description: str
    prerequisites: List[str]
    skills_required: List[str]
    estimated_time_hours: float
    resources: List[Dict[str, str]]
    status: str
    difficulty: str
    learning_style: List[str]
    outcome: str


def make_module_id(i: int) -> str:
    return f"MOD-{i:03d}"


def skill_to_module(skill: str) -> Tuple[str, str, str, str]:
    s = skill.lower()
    title = {
        "sql": "Practical SQL for Analytics & ML",
        "python": "Python for Data Science",
        "pandas": "Data Wrangling with Pandas",
        "numpy": "Numerical Computing with NumPy",
        "scikit-learn": "Applied Machine Learning with scikit-learn",
        "sklearn": "Applied Machine Learning with scikit-learn",
        "tensorflow": "Deep Learning with TensorFlow",
        "pytorch": "Deep Learning with PyTorch",
        "mlflow": "Experiment Tracking & Model Registry (MLflow)",
        "airflow": "Workflow Orchestration with Apache Airflow",
        "spark": "Distributed Data Processing with Spark",
        "docker": "Containerization with Docker",
        "kubernetes": "Kubernetes for ML & Data Workloads",
        "tableau": "Data Visualization with Tableau",
        "power bi": "Business Dashboards with Power BI",
        "xgboost": "Gradient Boosting with XGBoost",
        "dbt": "Analytics Engineering with dbt",
        "snowflake": "Modern Warehousing on Snowflake",
        "bigquery": "Google BigQuery for Analytics",
        "redshift": "Amazon Redshift for Analytics",
        "aws": "AWS for Data Scientists",
        "gcp": "GCP for Data Scientists",
        "azure": "Azure for Data Scientists",
        "s3": "Data Lakes on Amazon S3",
        "statistics": "Statistics & Experimentation Basics",
        "a/b testing": "A/B Testing & Causal Inference",
        "nlp": "Natural Language Processing Essentials",
        "computer vision": "Computer Vision Fundamentals",
        "time series": "Time-Series Analysis & Forecasting",
        "git": "Version Control with Git",
        "linux": "Linux & Bash for Data Workflows",
        "terraform": "Infrastructure as Code with Terraform",
        "databricks": "Lakehouse Workflows on Databricks",
        "prefect": "Dataflow Orchestration with Prefect",
    }.get(s, f"Mastering {skill.title()}")

    desc = f"Learn {skill} from the ground up with hands-on exercises mapped to common job requirements."
    outcome = f"Confidently apply {skill} in real job tasks."

    # heuristic difficulty
    difficulty = "beginner"
    if s in {"kubernetes", "spark", "terraform", "mlflow", "databricks", "nlp", "computer vision"}:
        difficulty = "intermediate"
    if s in {"kubernetes", "spark"}:
        difficulty = "advanced"

    return title, desc, outcome, difficulty


def build_modules(top_skills: List[Tuple[str, int]]) -> List[Module]:
    modules: List[Module] = []
    for i, (skill, score) in enumerate(top_skills, start=1):
        title, desc, outcome, difficulty = skill_to_module(skill)
        mod = Module(
            id=make_module_id(i),
            title=title,
            description=desc,
            prerequisites=["SKILL:Python basics"] if skill.lower() not in {"python", "git", "linux"} else [],
            skills_required=[skill],
            estimated_time_hours=2.0 if difficulty == "beginner" else (3.0 if difficulty == "intermediate" else 4.0),
            resources=[
                {"type": "article", "label": f"{title} – Guide", "url": "<add>"},
                {"type": "video", "label": f"{title} – Walkthrough", "url": "<add>"},
            ],
            status="not started",
            difficulty=difficulty,
            learning_style=["hands-on", "reading"],
            outcome=outcome,
        )
        modules.append(mod)
    return modules


# -----------------------------
# Utility: write outputs
# -----------------------------

def save_json(mods: List[Module], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in mods], f, ensure_ascii=False, indent=2)


def save_markdown(mods: List[Module], path: str) -> None:
    headers = [
        "ID", "Name / Title", "Description", "Prerequisites", "Est. Time", "Resources", "Status", "Difficulty", "Learning Style", "Outcome"
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for m in mods:
        res = ", ".join([r.get("label", "link") for r in m.resources])
        prereq = ", ".join(m.prerequisites) if m.prerequisites else "—"
        ls = ", ".join(m.learning_style)
        lines.append(
            "| " + " | ".join([
                m.id,
                m.title,
                m.description,
                prereq,
                f"{m.estimated_time_hours:g}h",
                res,
                m.status,
                m.difficulty,
                ls,
                m.outcome,
            ]) + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------

def main():
    greenhouse_org_list = ['roblox', 'kaizengaming', 'epicgames', 'Taboola', 'mangodb', 'esri', 'ensono', 'coinbase', 'databricks']
    lever_org_list = ['voleon', 'valence', 'gauntlet', 'weride', 'aquabyte', 'Until', 'binance', 'aircall', 'nava', 'BestEgg']
    ashby_org_list = ['notion', 'tandem', 'ziplines', 'atticus', 'deepL', 'sardine', 'mercor', 'hang', 'perplexity']
    
    parser = argparse.ArgumentParser(description="Generate top-15 modules from job skills.")
    parser.add_argument("--queries", nargs="+", default=["Data Scientist", "Machine Learning Engineer", "Algorithms Engineer"], help="Job title queries")
    parser.add_argument("--greenhouse-orgs", nargs="*", default=greenhouse_org_list, help="Greenhouse org handles (e.g., 'openai')")
    parser.add_argument("--lever-orgs", nargs="*", default=lever_org_list, help="Lever org handles (e.g., 'stripe')")
    parser.add_argument("--ashby-orgs", nargs="*", default=ashby_org_list, help="Ashby org handles (e.g., 'anthropic')")
    parser.add_argument("--limit-per-source", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--extra-skill", nargs="*", default=[], help="Add custom skills to track (phrases allowed)")
    parser.add_argument("--profile-text", type=str, default=None, help="Path to a text file with your profile content (optional)")

    args = parser.parse_args()

    collectors = []
    if args.greenhouse_orgs:
        collectors.append(GreenhouseCollector(args.greenhouse_orgs, args.limit_per_source))
    if args.lever_orgs:
        collectors.append(LeverCollector(args.lever_orgs, args.limit_per_source))
    if args.ashby_orgs:
        collectors.append(AshbyCollector(args.ashby_orgs, args.limit_per_source))

    if not collectors:
        print("No collectors configured. Provide at least one of --greenhouse-orgs/--lever-orgs/--ashby-orgs.")
        sys.exit(1)

    # Fetch job descriptions
    print("Collecting job descriptions…")
    jds: List[str] = []
    postings_meta: List[Dict[str, Any]] = []
    for c in collectors:
        batch = c.fetch(args.queries)
        for item in batch:
            text = clean_html_to_text(item.get("description_html", ""))
            if text:
                jds.append(text)
                postings_meta.append(item)
    print(f"Collected {len(jds)} job descriptions across sources.")

    # Add profile text if provided
    if args.profile_text and os.path.exists(args.profile_text):
        with open(args.profile_text, "r", encoding="utf-8", errors="ignore") as f:
            prof = f.read()
            jds.append(prof.lower())
            print("Included profile text for personalization.")

    # Extract skills
    print("Extracting skills…")
    counts = extract_skills(jds, args.extra_skill)
    if not counts:
        print("No skills detected. Try adding --extra-skill entries or more orgs.")
        sys.exit(2)

    # Rank skills (frequency). You can experiment with tf-idf weighting later.
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    # Pick top-k, but ensure variety (collapse synonyms)
    seen_roots = set()
    top: List[Tuple[str, int]] = []
    SYNONYMS = {
        "scikit-learn": {"sklearn"},
        "power bi": {"powerbi"},
    }
    def root_of(s: str) -> str:
        for root, syns in SYNONYMS.items():
            if s == root or s in syns:
                return root
        return s

    for skill, freq in ranked:
        r = root_of(skill)
        if r in seen_roots:
            continue
        seen_roots.add(r)
        top.append((r, freq))
        if len(top) >= args.top_k:
            break

    # Build modules
    modules = build_modules(top)

    # Save outputs
    save_json(modules, "modules_catalog.json")
    save_markdown(modules, "modules_catalog.md")

    print("\nTop skills:")
    for s, f in top:
        print(f"  - {s}: {f}")

    print("\nWrote: modules_catalog.json, modules_catalog.md")


if __name__ == "__main__":
    main()
