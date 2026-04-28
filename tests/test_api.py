from fastapi.testclient import TestClient

from api import app


def test_workflow_build_endpoint_runs_to_skeleton():
    client = TestClient(app)

    response = client.post(
        "/workflow/build",
        json={"text": "读取文章并生成摘要", "stage": "skeleton"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["stage"] == "skeleton"
    assert payload["skeleton"]["type"] == "Sequential"
    assert payload["dsl_output"] is None


def test_workflow_build_endpoint_rejects_unknown_stage():
    client = TestClient(app)

    response = client.post(
        "/workflow/build",
        json={"text": "读取文章并生成摘要", "stage": "unknown"},
    )

    assert response.status_code == 422
