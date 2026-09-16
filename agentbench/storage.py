import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager


class Store:
    def __init__(self, path="data/experiments.sqlite"):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created TEXT, mode TEXT, payload TEXT)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, experiment):
        with self.connect() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?)",
                       (experiment["id"], experiment["created"], experiment["mode"], json.dumps(experiment, allow_nan=False)))

    def history(self):
        with self.connect() as db:
            return [dict(id=r[0], created=r[1], mode=r[2]) for r in db.execute("SELECT id, created, mode FROM experiments ORDER BY created DESC")]

    def load(self, experiment_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if not row:
            raise KeyError(experiment_id)
        return json.loads(row[0])
