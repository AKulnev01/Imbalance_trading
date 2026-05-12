# utils/allocator.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import math
import datetime as dt

_CENTS = 2  # округление до центов

def _round2(x: float) -> float:
    return round(float(x or 0.0), _CENTS)

@dataclass
class _OpenReserve:
    usd_alloc: float
    created_at: dt.datetime
    est_close_at: Optional[dt.datetime] = None

class SmartAllocator:
    """
    Простой, предсказуемый аллокатор:
      • total_equity — текущий капитал (только реализованный PnL).
      • reserves     — сумма зарезервированных USD под открытые/ожидающие сделки.
      • fraction     — доля equity на СЛЕДУЮЩУЮ сделку (например 0.25 = 25%).
      • free_cash    = max(total_equity - sum(reserves), 0).

    Логика:
      • На новую сделку выдаём alloc = round(total_equity * fraction, 2).
      • Если free_cash < alloc → отдаём 0 (капитал-гейт → SKIP).
      • Закрытие сделки: close_one(now, usd_alloc, pnl_pct)
          -> снимаем резерв usd_alloc, применяем PnL = usd_alloc * (pnl_pct/100),
             обновляем total_equity += pnl_usd.
      • Отмена/истечение лимитки: close_one(..., pnl_pct=0.0)
    """
    def __init__(self, initial_total: float, fraction: float = 0.25):
        self.total_equity: float = _round2(initial_total)
        self.fraction: float = float(fraction)
        self._reserves: List[_OpenReserve] = []

    # ---- state ----
    def set_total(self, total: float) -> None:
        """Жёстко синкнемся с реальным балансом аккаунта (если используешь balance_sync)."""
        self.total_equity = _round2(total)

    def set_fraction(self, fraction: float) -> None:
        """Сменить долю на сделку (например 0.25)."""
        self.fraction = max(0.0, float(fraction))

    # ---- derived ----
    def reserves_sum(self) -> float:
        return _round2(sum(r.usd_alloc for r in self._reserves))

    def free_cash(self) -> float:
        fc = self.total_equity - self.reserves_sum()
        return _round2(fc if fc > 0 else 0.0)

    # ---- core API (совместимо с твоим кодом) ----
    def allocate_for_batch(
        self,
        batch_time: dt.datetime,
        batch_entries: List[dict],
        future_entries: List[dict]
    ) -> List[float]:
        """
        На каждый entry выдаём либо alloc=equity*fraction (округл. до центов), либо 0 если не хватает free_cash.
        batch_entries[i] может содержать 'close_time' — положим в метаданные (чисто информативно).
        """
        out: List[float] = []
        for ent in (batch_entries or []):
            wish = _round2(self.total_equity * self.fraction)
            if wish <= 0:
                out.append(0.0)
                continue
            if self.free_cash() >= wish:
                # резервируем
                self._reserves.append(_OpenReserve(
                    usd_alloc=wish,
                    created_at=batch_time,
                    est_close_at=ent.get("close_time") if isinstance(ent, dict) else None
                ))
                out.append(wish)
            else:
                out.append(0.0)
        return out

    def close_one(self, now: dt.datetime, usd_alloc: float, pnl_pct: float) -> None:
        """
        Снять один резерв на сумму usd_alloc и применить PnL по проценту.
        Используется:
          • при отмене лимитки/ошибке — pnl_pct=0
          • при закрытии позиции — pnl_pct = +3.0% или -1.0% (или что дал реальный исход)
        """
        usd_alloc = _round2(usd_alloc)
        # снять соответствующий резерв (по сумме). Если одинаковых несколько — снимем первый найденный.
        idx = None
        for i, r in enumerate(self._reserves):
            if _round2(r.usd_alloc) == usd_alloc:
                idx = i
                break
        if idx is None:
            # если по какой-то причине нет точного совпадения (например из-за округления),
            # снимем ближайший по величине резерв, чтобы не зависнуть.
            if self._reserves:
                idx = min(range(len(self._reserves)), key=lambda k: abs(self._reserves[k].usd_alloc - usd_alloc))
            else:
                # нечего снимать — просто применим PnL к total (fallback)
                pnl_usd = _round2(usd_alloc * (float(pnl_pct) / 100.0))
                self.total_equity = _round2(self.total_equity + pnl_usd)
                return

        reserve = self._reserves.pop(idx)
        # применяем PnL
        pnl_usd = _round2(reserve.usd_alloc * (float(pnl_pct) / 100.0))
        self.total_equity = _round2(self.total_equity + pnl_usd)

    # удобные геттеры для логов/мониторинга
    def snapshot(self) -> dict:
        return {
            "equity": self.total_equity,
            "reserves": self.reserves_sum(),
            "free_cash": self.free_cash(),
            "open_slots": len(self._reserves),
            "fraction": self.fraction,
        }