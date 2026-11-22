"""Local LSA-based retrieval and personalized hybrid re-ranking.
Only standard scientific Python dependencies; runs fully offline.
"""
import math
import sqlite3
from pathlib import Path
from typing import Any
import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import make_jobs

DB_PATH = Path(__file__).with_name("jobs.sqlite")


def init_db(db_path=DB_PATH):
    """Create a real SQLite table and populate it with synthetic demonstration jobs."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            company TEXT NOT NULL,
            title TEXT NOT NULL,
            track TEXT NOT NULL,
            skills_json TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT NOT NULL,
            level TEXT NOT NULL
        )""")
        if conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(r["id"], r["company"], r["title"], r["track"],
                  json.dumps(r["skills"]), r["description"], r["location"], r["level"])
                 for r in make_jobs()]
            )
            conn.commit()


def load_jobs(db_path=DB_PATH):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    result = []
    for row in rows:
        record = dict(row)
        record["skills"] = json.loads(record.pop("skills_json"))
        result.append(record)
    return result


class JobRecommender:
    def __init__(self, jobs):
        self.jobs = jobs
        corpus = [self._job_text(job) for job in jobs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        tfidf = self.vectorizer.fit_transform(corpus)
        dimensions = min(35, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        self.reducer = TruncatedSVD(n_components=dimensions, random_state=42)
        self.job_vectors = self.reducer.fit_transform(tfidf)
        self.job_tfidf = tfidf

    @staticmethod
    def _job_text(job):
        return (job["title"] + " " + job["description"] + " "
                + " ".join(job["skills"]) * 2)

    def recommend(self, profile: dict[str, Any], k=10, mode="hybrid"):
        query = profile["query"] + " " + " ".join(profile.get("skills", []))
        vector = self.vectorizer.transform([query])
        # A lexical baseline and a compact, learned dense LSA representation.
        lexical = cosine_similarity(vector, self.job_tfidf)[0]
        dense = self.reducer.transform(vector)
        semantic = cosine_similarity(dense, self.job_vectors)[0]
        max_semantic = max(float(np.max(semantic)), 1e-8)
        normalized_semantic = np.maximum(semantic, 0) / max_semantic
        desired = set(s.lower() for s in profile.get("skills", []))
        output = []
        for idx, job in enumerate(self.jobs):
            job_skills = set(job["skills"])
            overlap = len(desired & job_skills) / max(len(desired), 1)
            is_remote = job["location"] == "Remote"
            loc_match = job["location"] == profile.get("location")
            loc_score = 1.0 if loc_match or (is_remote and profile.get("remote_ok")) else 0.0
            level_score = 1.0 if job["level"] == profile.get("level", "Internship") else 0.0
            score = (float(lexical[idx]) if mode == "baseline" else
                     0.48 * float(normalized_semantic[idx]) + 0.32 * overlap +
                     0.12 * loc_score + 0.08 * level_score)
            output.append({
                "id": job["id"], "title": job["title"], "company": job["company"],
                "location": job["location"], "level": job["level"],
                "skills": job["skills"], "track": job["track"],
                "score": round(score, 5),
                "explanation": {
                    "shared_skills": sorted(desired & job_skills),
                    "preferred_location": bool(loc_score),
                    "experience_level_match": bool(level_score),
                }
            })
        return sorted(output, key=lambda x: (-x["score"], x["id"]))[:k]
