from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_healthz_endpoint():
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_endpoint():
    app = FastAPI()

    @app.get("/readyz")
    def readyz():
        return {"status": "ready"}

    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
