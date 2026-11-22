"""Offline evaluation over *synthetic* role-based preference labels.
Important: reported values do not measure real-world user satisfaction.
"""
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import ndcg_score

from data import TRACKS, make_profiles
from recommender import JobRecommender, load_jobs


def graded_label(job, profile):
    relevant = job["track"] == profile["target_track"]
    related = job["track"] in TRACKS[profile["target_track"]]["related"]
    location_match = (job["location"] == profile["location"] or
                      (job["location"] == "Remote" and profile["remote_ok"]))
    correct_level = job["level"] == profile["level"]
    if relevant:
        return 3 if location_match and correct_level else 2
    if related and location_match and correct_level:
        return 1
    return 0


def evaluate():
    jobs = load_jobs()
    recommender = JobRecommender(jobs)
    profiles = make_profiles()
    by_id = {j["id"]: j for j in jobs}
    report = {}
    for mode in ["baseline", "hybrid"]:
        ndcgs, recalls, reciprocal_ranks, latencies = [], [], [], []
        for p in profiles:
            t = time.perf_counter()
            recommendations = recommender.recommend(p, k=len(jobs), mode=mode)
            latencies.append((time.perf_counter()-t)*1000)
            truth = [graded_label(by_id[j["id"]], p) for j in recommendations]
            # ndcg_score receives the full ranking and computes DCG@10.
            ndcgs.append(float(ndcg_score([truth], [list(range(len(truth), 0, -1))], k=10)))
            relevant = {j["id"] for j in jobs if graded_label(j, p) >= 2}
            top10 = [j["id"] for j in recommendations[:10]]
            recalls.append(len(relevant & set(top10)) / max(len(relevant), 1))
            first = next((i for i, job_id in enumerate(top10, start=1) if job_id in relevant), None)
            reciprocal_ranks.append(1 / first if first else 0)
        report[mode] = {
            "NDCG@10": round(float(np.mean(ndcgs)), 4),
            "Recall@10": round(float(np.mean(recalls)), 4),
            "MRR@10": round(float(np.mean(reciprocal_ranks)), 4),
            "median_ranking_latency_ms": round(float(np.median(latencies)), 2),
        }
    report["benchmark"] = {
        "jobs": len(jobs), "profiles": len(profiles),
        "data_source": "fully synthetic role postings and synthetic relevance labels",
        "limitation": "Not a production benchmark or evidence of real-user relevance",
    }
    Path(__file__).with_name("benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
