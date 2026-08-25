from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _safe_run(runs_directory: Path, run_id: str) -> Path:
    root = runs_directory.resolve()
    candidate = (root / run_id).resolve()
    if (
        candidate == root
        or root not in candidate.parents
        or not (candidate / "config.json").exists()
    ):
        raise HTTPException(status_code=404, detail="run not found")
    return candidate


def create_app(runs_directory: Path = Path("runs")) -> FastAPI:
    app = FastAPI(title="Contrast Lab", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        found = []
        if not runs_directory.exists():
            return found
        for config_path in runs_directory.glob("*/*/config.json"):
            directory = config_path.parent
            events = _metrics(directory / "metrics.jsonl")
            config = _read_json(config_path)
            found.append(
                {
                    "id": str(directory.relative_to(runs_directory)),
                    "experiment": config["run"]["experiment"],
                    "seed": config["run"]["seed"],
                    "objective": config["objective"]["kind"],
                    "strategy": config["training"]["step_strategy"],
                    "latest": events[-1] if events else None,
                }
            )
        return sorted(found, key=lambda item: item["id"], reverse=True)

    @app.get("/api/run/{run_id:path}")
    def run(run_id: str) -> dict[str, Any]:
        directory = _safe_run(runs_directory, run_id)
        environment_path = directory / "environment.json"
        return {
            "id": run_id,
            "config": _read_json(directory / "config.json"),
            "environment": _read_json(environment_path) if environment_path.exists() else {},
            "metrics": _metrics(directory / "metrics.jsonl"),
        }

    distribution = Path(__file__).parents[3] / "web" / "dist"
    if distribution.exists():
        assets = distribution / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            requested = distribution / path
            if (
                path
                and requested.is_file()
                and distribution.resolve() in requested.resolve().parents
            ):
                return FileResponse(requested)
            return FileResponse(distribution / "index.html")

    return app
