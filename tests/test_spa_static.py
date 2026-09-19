from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import SpaStaticFiles


def _spa_client(tmp_path: Path) -> tuple[TestClient, str, str]:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    index_marker = "loreguard-spa-index"
    asset_marker = "loreguard-static-asset"
    (dist / "index.html").write_text(index_marker, encoding="utf-8")
    (assets / "app.js").write_text(asset_marker, encoding="utf-8")

    test_app = FastAPI()

    @test_app.get("/api/v1/existing")
    def existing_api() -> dict[str, bool]:
        return {"ok": True}

    test_app.mount("/", SpaStaticFiles(directory=dist, html=True), name="web")
    return TestClient(test_app), index_marker, asset_marker


def test_spa_routes_serve_index_for_direct_navigation(tmp_path: Path) -> None:
    client, index_marker, _ = _spa_client(tmp_path)
    routes = [
        "/",
        "/login",
        "/register",
        "/app",
        "/app/projects/project-id/check",
        "/app/projects/project-id/documents/chapter-1",
        "/app/settings/model",
        "/check",
        "/projects",
        "/diff",
        "/visual",
        "/audit",
        "/report",
        "/provider",
    ]

    with client:
        for route in routes:
            response = client.get(route)
            assert response.status_code == 200, route
            assert response.text == index_marker, route
            assert response.headers["content-type"].startswith("text/html"), route


def test_real_assets_and_registered_api_routes_keep_priority(tmp_path: Path) -> None:
    client, index_marker, asset_marker = _spa_client(tmp_path)

    with client:
        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert asset.text == asset_marker
        assert asset.text != index_marker

        api = client.get("/api/v1/existing")
        assert api.status_code == 200
        assert api.json() == {"ok": True}


def test_unknown_api_and_missing_resources_are_not_swallowed(tmp_path: Path) -> None:
    client, index_marker, _ = _spa_client(tmp_path)

    with client:
        for route in (
            "/api/v1/not-a-route",
            "/api/not-a-route",
            "/assets/missing.js",
            "/assets/missing",
            "/app/missing.js",
            "/favicon.ico",
            "/unknown",
        ):
            response = client.get(route)
            assert response.status_code == 404, route
            assert response.text != index_marker, route


def test_directory_traversal_attempts_cannot_read_outside_dist(tmp_path: Path) -> None:
    client, index_marker, _ = _spa_client(tmp_path)
    secret_marker = "must-not-be-served"
    (tmp_path / "secret.txt").write_text(secret_marker, encoding="utf-8")

    with client:
        for route in (
            "/%2e%2e/secret.txt",
            "/app/%2e%2e/%2e%2e/secret.txt",
            "/app/%5c..%5csecret.txt",
            "/app/%2e%2e/app",
        ):
            response = client.get(route)
            assert response.status_code == 404, route
            assert response.text != index_marker, route
            assert secret_marker not in response.text, route
