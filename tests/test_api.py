import sqlite3
from datetime import date

from fastapi.testclient import TestClient
from arxiv_reader.api import KEPT, REJECTED, create_app
from conftest import open_mem, run

DAY = date(2026, 10, 5)


def weight(tmp_path, paper_id):
    row = sqlite3.connect(tmp_path / "memory.db").execute(
        "SELECT weight FROM events WHERE item_id = ?", (paper_id,)).fetchone()
    return row and row[0]


def app(tmp_path, app_db):
    # The memory is opened inside the app, on the server thread, as `serve` does.
    return create_app(lambda: app_db, lambda: open_mem(tmp_path))


def test_feedback_writes_weights(tmp_path, app_db, mem, papers):
    run(app_db, mem, DAY, papers)
    with TestClient(app(tmp_path, app_db)) as client:
        digest = client.get("/digest/today").json()
        first, second = digest["items"][0]["id"], digest["items"][1]["id"]
        assert len(digest["items"]) == 5 and digest["stale"] == (date.today().weekday() < 5)

        assert client.put(f"/papers/{first}/not-interested").status_code == 200
        assert weight(tmp_path, first) == REJECTED
        assert client.put(f"/digest/{DAY}/opened").status_code == 200
        assert weight(tmp_path, first) == REJECTED  # opening does not overwrite a rejection
        assert weight(tmp_path, second) == KEPT

        assert client.delete(f"/papers/{first}/not-interested").status_code == 200
        assert weight(tmp_path, first) == KEPT
        assert client.get("/digest/today").json()["items"][0]["rejected"] is False
        assert client.put("/papers/nope/not-interested").status_code == 404


def test_undo_before_opening_forgets(tmp_path, app_db, mem, papers):
    run(app_db, mem, DAY, papers)
    with TestClient(app(tmp_path, app_db)) as client:
        client.put("/papers/p0/not-interested")
        client.delete("/papers/p0/not-interested")
        assert weight(tmp_path, "p0") is None
