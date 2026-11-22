# JobMatch — Personalized Job Recommendations (Prototype)

A **working, offline demonstration** of an SDE/AI internship-relevant job recommendation API. Includes synthetic listings, a SQLite database, TF-IDF and low-rank latent semantic analysis (LSA) retrieval, skill-aware and location-aware re-ranking, explanations, and a repeatable offline evaluation using **NDCG@10, Recall@10, and MRR@10**.

**Data transparency:** All 72 jobs, 18 example candidate profiles, company names, and relevance labels are **synthetic**. Benchmark scores only describe this toy benchmark—not production job relevance, real recruiters, or live users. This is a prototype, not a deployed product or a transformer-embedding system.

## Run locally

```bash
python -m pip install -r requirements.txt
python evaluate.py
pytest -q
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API documentation.

## Example API request

```bash
curl -X POST 'http://127.0.0.1:8000/recommend?k=5' \
  -H 'Content-Type: application/json' \
  -d '{"query":"AI agent engineering internship with Python, retrieval and LLM evaluation", "skills":["python","fastapi","rag","embeddings","sql"], "location":"Bengaluru", "remote_ok":true, "level":"Internship"}'
```

## Design

1. `data.py`: Create **72 synthetic job records** across nine job families, and **18 synthetic example candidates**.
2. `recommender.py`: Persist jobs to an actual **SQLite** table. Fit TF-IDF and `TruncatedSVD` to produce **35-dimensional LSA vectors**, ranking with a weighted combination of LSA similarity, skill overlap, location preference, and level compatibility. Compare with a TF-IDF-only baseline.
3. `app.py`: Expose `/health`, `/jobs`, and `/recommend` via **FastAPI**; return personalized ranking and matching explanations.
4. `evaluate.py`: Evaluate offline using graded synthetic relevance judgments and **NDCG@10, Recall@10, MRR@10**, plus median query-ranking latency. Output to `benchmark.json`.
5. `tests/`: Smoke tests for API, seed count, ranking determinism, and benchmark bounds.

## Evaluation caveats

The benchmark reuses the same job-family vocabulary across synthetic descriptions and candidates; metrics are **optimistic** and are not evidence of generalization. Timing is local, includes Python ranking but excludes network, model loading, and concurrent load. The next credible step is an external dataset with human annotations, plus actual transformer embeddings, retraining and data-refresh logic, user feedback, held-out evaluation, and deployment/observability.

## Using this project in a resume

Use only after you have run and inspected the project, modified or maintained it, and can explain the indexing and ranking/evaluation choices. Describe its scores as **synthetic offline benchmarks**. Do not claim real-world adoption, deployment, production metrics, or that it uses LLM agents.
