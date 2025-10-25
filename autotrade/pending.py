# autotrade/pending.py
import os, json, threading
from datetime import datetime, timezone, timedelta

class PendingEntryStore:
    def __init__(self, path="./state/pending_entries.json", grace_sec=120):
        self.path = path
        self.grace = int(os.getenv("ENTRY_GRACE_SEC", grace_sec))
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        # нормализуем
        self.data = {}
        for k, v in raw.items():
            try:
                v["imb_time"] = datetime.fromisoformat(v["imb_time"]).replace(tzinfo=timezone.utc)
                self.data[k] = v
            except Exception:
                continue

    def _save(self):
        out = {}
        for k, v in self.data.items():
            vv = dict(v)
            vv["imb_time"] = vv["imb_time"].astimezone(timezone.utc).isoformat().replace("+00:00","Z")
            out[k] = vv
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    @staticmethod
    def key(symbol: str, imb_time) -> str:
        t = imb_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"{symbol.upper()}::{t}"

    def add(self, symbol: str, imb_time, strength: float, side: str, detect_px: float, meta=None):
        with self._lock:
            k = self.key(symbol, imb_time)
            if k in self.data:  # уже в очереди
                return False
            self.data[k] = {
                "symbol": symbol.upper(),
                "imb_time": imb_time.astimezone(timezone.utc),
                "deadline": (imb_time + timedelta(seconds=self.grace)).astimezone(timezone.utc),
                "strength": float(strength),
                "side": side.upper(),
                "detect_px": float(detect_px),
                "status": "queued",
                "meta": meta or {}
            }
            self._save()
            return True

    def mark(self, key: str, status: str, reason: str = "", extra: dict = None):
        with self._lock:
            if key not in self.data:
                return
            self.data[key]["status"] = status
            if reason:
                self.data[key]["reason"] = reason
            if extra:
                self.data[key].update(extra)
            self._save()

    def remove(self, key: str):
        with self._lock:
            if key in self.data:
                self.data.pop(key)
                self._save()

    def list_active(self):
        now = datetime.now(timezone.utc)
        with self._lock:
            return {
                k: v for k, v in self.data.items()
                if v.get("status") in ("queued","retry") and now <= v["deadline"]
            }

    def list_expired(self):
        now = datetime.now(timezone.utc)
        with self._lock:
            return {
                k: v for k, v in self.data.items()
                if v.get("status") in ("queued","retry") and now > v["deadline"]
            }

    def snapshot(self):
        with self._lock:
            return dict(self.data)