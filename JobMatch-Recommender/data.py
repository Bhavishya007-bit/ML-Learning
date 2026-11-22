"""Synthetic data for an offline job-recommendation demonstration.
No employer listings or user profiles here are real.
"""

TRACKS = {
    "ai_agents": {
        "titles": ["AI Agent Engineer Intern", "Applied AI Intern", "LLM Applications Intern", "AI Software Engineer Intern"],
        "skills": ["python", "fastapi", "llm", "rag", "tool calling", "postgresql", "embeddings", "docker"],
        "description": "Build tool-using conversational agents, evaluate grounded answers, and expose reliable AI workflows through Python APIs.",
        "related": ["search_recs", "nlp", "backend"],
    },
    "search_recs": {
        "titles": ["Search Relevance Intern", "Recommendation Systems Intern", "Ranking Engineer Intern", "Personalization Intern"],
        "skills": ["python", "ranking", "ndcg", "retrieval", "embeddings", "sql", "scikit-learn", "personalization"],
        "description": "Improve search retrieval, build personalized recommendations, and measure ranking quality with NDCG and recall metrics.",
        "related": ["ai_agents", "ml", "nlp"],
    },
    "backend": {
        "titles": ["Python Backend Intern", "SDE Backend Intern", "API Engineering Intern", "Database Backend Intern"],
        "skills": ["python", "fastapi", "rest api", "sql", "postgresql", "docker", "testing", "redis"],
        "description": "Build REST APIs, maintain SQL data models, test services, and improve backend reliability and latency.",
        "related": ["ai_agents", "data_engineer"],
    },
    "ml": {
        "titles": ["Machine Learning Intern", "Applied ML Intern", "Modeling Intern", "ML Platform Intern"],
        "skills": ["python", "scikit-learn", "xgboost", "statistics", "feature engineering", "model evaluation", "numpy", "pandas"],
        "description": "Develop predictive models and feature pipelines, validate offline experiments, and evaluate model performance.",
        "related": ["search_recs", "nlp", "analytics"],
    },
    "nlp": {
        "titles": ["NLP Engineering Intern", "Language AI Intern", "Text Mining Intern", "Conversational AI Intern"],
        "skills": ["python", "nlp", "tokenization", "transformers", "embeddings", "classification", "pytorch", "llm"],
        "description": "Create text understanding systems, retrieval components, language-model evaluation and classification pipelines.",
        "related": ["ai_agents", "ml", "search_recs"],
    },
    "data_engineer": {
        "titles": ["Data Engineering Intern", "ETL Development Intern", "Database Systems Intern", "Analytics Engineering Intern"],
        "skills": ["python", "sql", "etl", "postgresql", "airflow", "dbt", "data modeling", "data validation"],
        "description": "Design ETL pipelines, SQL warehouse schemas and quality checks to process large structured datasets.",
        "related": ["backend", "analytics"],
    },
    "frontend": {
        "titles": ["Frontend React Intern", "Web UI Engineering Intern", "TypeScript Intern", "Web Product Intern"],
        "skills": ["typescript", "javascript", "react", "html", "css", "ui testing", "vite", "accessibility"],
        "description": "Implement responsive React interfaces, component tests, accessible layouts and product analytics events.",
        "related": ["backend"],
    },
    "devops": {
        "titles": ["Cloud Engineering Intern", "DevOps Intern", "Site Reliability Intern", "Platform Infrastructure Intern"],
        "skills": ["linux", "docker", "kubernetes", "aws", "terraform", "ci/cd", "monitoring", "python"],
        "description": "Automate cloud deployments, configure container orchestration and monitor service reliability.",
        "related": ["backend", "data_engineer"],
    },
    "analytics": {
        "titles": ["Data Analyst Intern", "Product Analytics Intern", "Business Intelligence Intern", "Decision Science Intern"],
        "skills": ["sql", "python", "power bi", "statistics", "excel", "pandas", "dashboards", "experimentation"],
        "description": "Build SQL reporting datasets, dashboards and experiments to support evidence-based product decisions.",
        "related": ["ml", "data_engineer"],
    },
}

LOCATIONS = ["Remote", "Bengaluru", "Gurugram", "Hyderabad", "Remote", "Pune", "Remote", "Delhi"]


def make_jobs():
    jobs = []
    idx = 1
    for track, meta in TRACKS.items():
        for i in range(8):
            skills = meta["skills"][:]
            # All rows are intentionally synthetic, with some variation in skill emphasis.
            if i % 3 == 0:
                skills = skills[:6]
            elif i % 3 == 1:
                skills = skills[2:]
            title = meta["titles"][i % 4]
            jobs.append({
                "id": idx,
                "company": f"Sample Employer {idx:02d}",
                "title": title,
                "track": track,
                "skills": skills,
                "description": meta["description"],
                "location": LOCATIONS[(i + list(TRACKS).index(track)) % len(LOCATIONS)],
                "level": "Internship" if i % 5 != 0 else "Junior",
            })
            idx += 1
    return jobs


def make_profiles():
    profiles = []
    for track, meta in TRACKS.items():
        for ix in range(2):
            skills = meta["skills"][ix:ix + 5]
            profiles.append({
                "name": f"demo_{track}_{ix}",
                "query": f"Seeking {meta['titles'][ix].replace(' Intern', '')} opportunities. " + meta["description"],
                "skills": skills,
                "location": "Bengaluru" if ix == 0 else "Gurugram",
                "remote_ok": ix == 1,
                "level": "Internship",
                "target_track": track,
            })
    return profiles
