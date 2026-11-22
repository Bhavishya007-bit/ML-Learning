from fastapi.testclient import TestClient
from app import app
from data import make_jobs, make_profiles
from evaluate import evaluate
from recommender import JobRecommender


def test_demo_fixture_counts():
    assert len(make_jobs()) == 72
    assert len(make_profiles()) == 18


def test_api_health_and_recommendations():
    client = TestClient(app)
    assert client.get('/health').json()['job_count'] == 72
    response = client.post('/recommend?k=5', json={
        'query': 'Python backend and API development',
        'skills': ['python', 'fastapi', 'sql'],
        'location': 'Bengaluru', 'remote_ok': True, 'level': 'Internship'
    })
    assert response.status_code == 200
    assert response.json()['count'] == 5
    assert 'explanation' in response.json()['recommendations'][0]


def test_ranking_is_deterministic():
    engine = JobRecommender(make_jobs())
    profile = make_profiles()[0]
    assert engine.recommend(profile) == engine.recommend(profile)


def test_benchmark_outputs_metrics():
    data = evaluate()
    assert 0 <= data['hybrid']['NDCG@10'] <= 1
    assert 0 <= data['hybrid']['Recall@10'] <= 1
    assert 0 <= data['hybrid']['MRR@10'] <= 1
