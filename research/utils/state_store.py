import json
import os
from typing import Dict, Any

class JsonState:
    """
    Хранилище состояния в JSON:
      - processed_ids: уже обработанные сигналы (чтобы не повторять вход)
      - open_positions: открытые позиции (oid -> данные)
      - kv: универсальный словарь для произвольного использования
             (например, pending-лимитки с TTL)
    """

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = {
            "processed_ids": [],
            "open_positions": {},
            "kv": {}
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # --- Универсальный KV (используется для pending и пр.) ---
    def get(self, key: str, default=None):
        return self._data.get("kv", {}).get(key, default)

    def upsert(self, key: str, value: Any):
        self._data.setdefault("kv", {})[key] = value
        self._save()

    def pop(self, key: str, default=None):
        val = self._data.setdefault("kv", {}).pop(key, default)
        self._save()
        return val

    def all(self) -> Dict[str, Any]:
        # возвращаем копию, чтобы внешний код не менял напрямую внутреннее состояние
        return dict(self._data.get("kv", {}))

    # --- Методы для сигналов ---
    def is_processed(self, oid: str) -> bool:
        return oid in self._data["processed_ids"]

    def mark_processed(self, oid: str):
        if oid not in self._data["processed_ids"]:
            self._data["processed_ids"].append(oid)
            self._save()

    # --- Методы для открытых позиций ---
    def upsert_open(self, oid: str, info: Dict[str, Any]):
        self._data["open_positions"][oid] = info
        self._save()

    def all_open(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._data.get("open_positions", {}))

    def pop_open(self, oid: str):
        if oid in self._data["open_positions"]:
            self._data["open_positions"].pop(oid)
            self._save()