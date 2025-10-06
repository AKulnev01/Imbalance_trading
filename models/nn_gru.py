import os, json, math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# GRU, но работаем и с seq_len=1 (если фичи агрегированные).
# Вход = последовательность по фичам рынка (fx), плюс статичный вектор θ (th), который конкатим после GRU.

class GRUSelector(nn.Module):
    def __init__(self, fx_dim: int, th_dim: int, hidden: int = 128, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=fx_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + th_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_th: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, fx_dim], x_th: [B, th_dim]
        out, _ = self.gru(x_seq)      # [B, T, hidden]
        h = out[:, -1, :]             # последний таймстеп
        z = torch.cat([h, x_th], dim=1)
        y = self.head(z)              # [B, 1]
        return y.squeeze(-1)

def _to_seq(X_fx: np.ndarray, seq_len: int) -> np.ndarray:
    # если у нас агрегированные фичи на момент входа — делаем фиктивную длину 1
    # если заранее заготовишь лаги, можешь собрать [B, T, fx_dim]
    if seq_len == 1:
        return X_fx[:, None, :]  # [B,1,F]
    raise ValueError("Сейчас поддержан только seq_len=1; добавь сбор lag-фичей для T>1.")

def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, q: float = 0.5) -> torch.Tensor:
    # Можно тренировать на медиане (q=0.5) как робастный таргет вместо MSE
    diff = y_true - y_pred
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))

def train_model(X_fx, X_th, y, out_path: str, seq_len: int = 1, epochs: int = 50, lr: float = 1e-3, device: str = "cpu"):
    os.makedirs(out_path, exist_ok=True)
    X_seq = _to_seq(X_fx, seq_len)
    X_seq = torch.tensor(X_seq, dtype=torch.float32, device=device)
    X_th  = torch.tensor(X_th,  dtype=torch.float32, device=device)
    y     = torch.tensor(y,     dtype=torch.float32, device=device)

    fx_dim = X_seq.shape[-1]
    th_dim = X_th.shape[-1]
    model = GRUSelector(fx_dim=fx_dim, th_dim=th_dim, hidden=128, layers=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 1e9
    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        y_pred = model(X_seq, X_th)
        # смешанный лосс: медианная pinball + L2
        loss = pinball_loss(y_pred, y, q=0.5) + 0.1 * torch.mean((y_pred - y) ** 2)
        loss.backward()
        opt.step()

        if loss.item() < best:
            best = loss.item()
            torch.save({
                "state_dict": model.state_dict(),
                "fx_dim": fx_dim,
                "th_dim": th_dim
            }, os.path.join(out_path, "gru_selector.pt"))
    meta = {"seq_len": seq_len, "epochs": epochs, "best_loss": float(best)}
    with open(os.path.join(out_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_model(model_dir: str, device: str = "cpu") -> Tuple[GRUSelector, dict]:
    ckpt = torch.load(os.path.join(model_dir, "gru_selector.pt"), map_location=device)
    model = GRUSelector(fx_dim=ckpt["fx_dim"], th_dim=ckpt["th_dim"], hidden=128, layers=1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def predict(model: GRUSelector, X_fx: np.ndarray, X_th: np.ndarray, device: str = "cpu", seq_len: int = 1) -> np.ndarray:
    with torch.no_grad():
        X_seq = _to_seq(X_fx, seq_len)
        X_seq = torch.tensor(X_seq, dtype=torch.float32, device=device)
        X_th  = torch.tensor(X_th,  dtype=torch.float32, device=device)
        y     = model(X_seq, X_th).cpu().numpy()
        return y

if __name__ == "__main__":
    import argparse
    from ml.nn_dataset import load_offline, split_columns, make_arrays, train_val_split_idx

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)           # offline.parquet (fx__, th__, pnl_pct_after_cost)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=1)  # сейчас 1; для >1 нужны лаги в фичах
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = load_offline(args.data)
    fx_cols, th_cols = split_columns(df)
    X_fx, X_th, y = make_arrays(df, fx_cols, th_cols)
    train_idx, val_idx = train_val_split_idx(len(df), val_frac=0.2)
    # тренируем на всем (коротко), можно легко расширить на батчи/валидацию
    train_model(X_fx, X_th, y, out_path=args.out_dir, seq_len=args.seq_len, epochs=args.epochs, lr=args.lr, device=args.device)