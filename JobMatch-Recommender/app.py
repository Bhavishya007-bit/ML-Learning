"""Run with: uvicorn app:app --reload"""
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from recommender import JobRecommender, load_jobs

app = FastAPI(title="JobMatch: Personalized Recommendation API", version="0.1.0")
recommender = JobRecommender(load_jobs())


class Candidate(BaseModel):
    query: str = Field(min_length=5)
    skills: list[str] = Field(default_factory=list)
    location: str = "Bengaluru"
    remote_ok: bool = True
    level: str = "Internship"


@app.get("/health")
def health():
    return {"status": "ok", "job_count": len(recommender.jobs)}


@app.get("/jobs")
def jobs(limit: int = Query(20, ge=1, le=100)):
    return recommender.jobs[:limit]


@app.post("/recommend")
def recommend(candidate: Candidate, k: int = Query(10, ge=1, le=50)):
    result = recommender.recommend(candidate.model_dump(), k=k)
    return {"count": len(result), "recommendations": result}
