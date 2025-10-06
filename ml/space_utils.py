import random, yaml
from typing import Dict, Any, List

def load_space(path_yaml: str) -> dict:
    with open(path_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sample_thetas(space: dict, limit: int = 4000) -> list:
    risk = space["risk"]; entry = space["entry"]; exit_ = space["exit"]; filters = space["filters"]
    combos = []
    for tp in risk["tp_rr"]:
        for sl in risk["sl_rr"]:
            if sl >= tp:
                continue
            for ttl in risk["ttl_hours"]:
                for cc in entry["confirm_candles_4h"]:
                    for mk in entry["mkt_order"]:
                        for ms in entry["max_slippage_bp"]:
                            for tt in exit_["trail_type"]:
                                for tm in exit_["trail_mult"]:
                                    for spr in filters["min_spread_bp"]:
                                        for liq in filters["min_liquidity_usd"]:
                                            for vlb in filters["volatility_lookback_h"]:
                                                for ses in filters["session"]:
                                                    combos.append({
                                                        "tp_rr": tp, "sl_rr": sl, "ttl_hours": ttl,
                                                        "confirm_candles_4h": cc, "mkt_order": mk, "max_slippage_bp": ms,
                                                        "trail_type": tt, "trail_mult": tm,
                                                        "min_spread_bp": spr, "min_liquidity_usd": liq,
                                                        "volatility_lookback_h": vlb, "session": ses
                                                    })
    random.shuffle(combos)
    return combos[:limit]