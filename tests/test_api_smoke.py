"""API-Smoke-Tests: Routen antworten, Validierung und Fehler-Mapping stimmen.

Bewusst ohne Modell-Laden/Training — die Tests laufen in Sekunden und ohne
Dateien in models/.
"""
from fastapi.testclient import TestClient

from pdl_lt_sg_predict.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_config_shape():
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["enable_training"], bool)
    codes = [m["code"] for m in data["model_types"]]
    assert "svm" in codes and "nn" in codes


def test_models_list():
    r = client.get("/api/models")
    assert r.status_code == 200


def test_predict_single_empty_bedeutung_422():
    r = client.post("/api/predict/single", json={
        "model_file": "egal.pkl", "lemma": "", "bedeutung": "   ",
    })
    assert r.status_code == 422


def test_predict_single_missing_model_404():
    # Fixiert das Fehler-Mapping: FileNotFoundError aus get_model() muss als
    # 404 ankommen, nicht als 500 (bridge.get_model reicht sie durch).
    r = client.post("/api/predict/single", json={
        "model_file": "gibt_es_nicht.pkl", "lemma": "", "bedeutung": "Haus",
    })
    assert r.status_code == 404


def test_predict_batch_missing_model_404():
    r = client.post(
        "/api/predict/batch",
        data={"model_file": "gibt_es_nicht.pkl"},
        files={"file": ("test.csv", b"lemma,bedeutung\nHaus,Gebaeude\n", "text/csv")},
    )
    assert r.status_code == 404


def test_predict_batch_rejects_non_csv():
    r = client.post(
        "/api/predict/batch",
        data={"model_file": "egal.pkl"},
        files={"file": ("test.txt", b"foo", "text/plain")},
    )
    assert r.status_code == 422


def test_anleitung_markdown():
    r = client.get("/api/anleitung")
    assert r.status_code == 200
    assert "markdown" in r.json()
