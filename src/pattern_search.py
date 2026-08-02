"""
패널 기반 패턴 탐색 — 30개 자산에서 동시에 통하는 캔들 형태를 찾는다.

기존 프로젝트와 무엇이 다른가
  기존: BTC 한 종목에서 사람이 형태를 착안 → BTC OOS로 검증 → 확정.
        이 절차는 BTC에만 맞는 형태를 고르게 되어 있고, 실제로 확정된 4종
        전부가 다른 29개 자산에서 랜덤 진입 대비 우위가 없었다.
  여기:  후보를 기계적으로 열거하고 30개 자산 IS 패널에서 고른 뒤,
        건드린 적 없는 OOS 패널에서 확인한다.

자체 창안 원칙 준수
  후보는 알려진 패턴 이름에서 오지 않는다. 캔들의 기하학적 성질
  (몸통비·꼬리비·종가위치·갭·방향·범위백분위)을 조합해 열거할 뿐이다.

과적합 방지
  후보를 수천 개 시험하므로 IS 최고 성적은 반드시 요행을 포함한다.
  그래서 IS는 "후보 압축" 용도로만 쓰고, 판정은 OOS 패널에서
  다중검정 보정된 문턱으로 내린다.

2단 평가
  1단(전수): 자산별 edge를 벡터 연산으로 계산 — 후보 하나당 수 ms
  2단(상위): 순환이동 검정 2,000회 — 후보 하나당 수 초
  1단만 수천 개 돌리고 2단은 상위 소수에만 적용한다.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import ASSET_CLASS
from panel_eval import MIN_ASSETS, MIN_TRADES, SEED, Panel, score
from multi_asset_validation import shift_test

RESULTS_DIR = Path(__file__).parent.parent / "results"
WINDOW = 60


# ──────────────────────────────────────────────
# 특징량 — 전부 비율/순위. 절대 가격 수준은 쓰지 않는다 (원칙 4-2-1).
# ──────────────────────────────────────────────
def features(df: pd.DataFrame) -> dict:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rng = np.where(h - l > 0, h - l, np.nan)
    body = np.abs(c - o)
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - l

    def pctrank(x):
        """직전 WINDOW일 안에서의 순위(0~1). 자산 간 스케일을 없애기 위함."""
        s = pd.Series(x)
        return (s.rolling(WINDOW)
                 .apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
                 .to_numpy())

    f = {
        "body":      body / rng,            # 몸통 비율
        "upper":     up / rng,              # 위꼬리 비율
        "lower":     dn / rng,              # 아래꼬리 비율
        "clspos":    (c - l) / rng,         # 종가 위치
        "opnpos":    (o - l) / rng,         # 시가 위치
        "bull":      (c > o).astype(float),  # 양봉 여부
        "rngrank":   pctrank(h - l),        # 범위의 최근 60일 순위
    }
    prev_rng = np.concatenate([[np.nan], rng[:-1]])
    prev_c = np.concatenate([[np.nan], c[:-1]])
    f["gap"] = (o - prev_c) / prev_rng      # 갭 (전일 범위 대비)
    f["chg"] = (c - prev_c) / prev_rng      # 종가 변화 (전일 범위 대비)
    return f


# 조건 원자: (특징, 지연, 방향, 문턱)
# 지연 1~3 = 어제·그제·그끄제. 오늘(지연 0) 값은 조건에 쓰지 않는다 —
# 오늘 봉이 완성되어야 알 수 있는 값으로 오늘 진입하면 실행 불가능하기 때문.
# 오늘 정보는 아래 TRIGGERS에서 종가 확정 후 판정 가능한 형태로만 쓴다.
ATOMS = []
for feat, thresholds in [
    ("body",   [0.3, 0.5, 0.7]),
    ("upper",  [0.2, 0.35, 0.5]),
    ("lower",  [0.2, 0.35, 0.5]),
    ("clspos", [0.3, 0.5, 0.7, 0.85]),
    ("opnpos", [0.3, 0.5, 0.7]),
    ("rngrank", [0.3, 0.5, 0.8]),
    ("gap",    [-0.2, 0.0, 0.2]),
    ("chg",    [-0.5, 0.0, 0.5]),
]:
    for lag in (1, 2, 3):
        for thr in thresholds:
            for op in (">=", "<="):
                ATOMS.append((feat, lag, op, thr))
for lag in (1, 2, 3):
    ATOMS.append(("bull", lag, ">=", 0.5))    # 양봉
    ATOMS.append(("bull", lag, "<=", 0.5))    # 음봉


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if k > 0:
        out[k:] = a[:-k]
    return out


def atom_mask(f: dict, atom) -> np.ndarray:
    feat, lag, op, thr = atom
    v = _shift(f[feat], lag)
    with np.errstate(invalid="ignore"):
        m = v >= thr if op == ">=" else v <= thr
    return np.where(np.isnan(v), False, m)


# 오늘의 방아쇠 — 종가가 확정된 뒤 판정하고 다음날 시가에 진입하므로 실행 가능하다.
TRIGGERS = {
    "close>prev_high":  lambda df: df["close"].to_numpy() > _shift(df["high"].to_numpy(), 1),
    "close>prev_close": lambda df: df["close"].to_numpy() > _shift(df["close"].to_numpy(), 1),
    "close<prev_low":   lambda df: df["close"].to_numpy() < _shift(df["low"].to_numpy(), 1),
    "close>high3":      lambda df: df["close"].to_numpy() > pd.Series(df["high"]).shift(1).rolling(3).max().to_numpy(),
    "none":             lambda df: np.ones(len(df), dtype=bool),
}


class SearchPanel(Panel):
    """Panel에 특징량·방아쇠 캐시를 얹어 후보 하나당 비용을 배열 AND 수준으로 낮춘다."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.feat, self.trig = {}, {}
        for name, df in self.data.items():
            self.feat[name] = features(df)
            self.trig[name] = {k: np.nan_to_num(fn(df), nan=0).astype(bool)
                               for k, fn in TRIGGERS.items()}
            # 랜덤 진입 기준선을 구간별로 한 번만 계산해 둔다
            ex = self.exits[name]
            for per, mask in (("IS", self.is_mask[name]), ("OOS", self.oos_mask[name])):
                pool = ex[mask & ~np.isnan(ex)]
                setattr(self, f"_base_{per}_{name}",
                        float(pool.mean()) if len(pool) else np.nan)

    def signal(self, name: str, atoms, trigger: str) -> np.ndarray:
        f = self.feat[name]
        m = self.trig[name][trigger].copy()
        for a in atoms:
            m &= atom_mask(f, a)
        return m

    def quick_score(self, atoms, trigger: str, period: str) -> dict:
        """1단 스크리닝 — 자산별 edge만 계산. 순환이동 검정 없음."""
        masks = self.is_mask if period == "IS" else self.oos_mask
        edges, trades, n_ok = [], 0, 0
        cls_pos = {"crypto": [], "equity": [], "commodity_fx": []}
        for name in self.data:
            ex = self.exits[name]
            sel = self.signal(name, atoms, trigger) & masks[name] & ~np.isnan(ex)
            k = int(sel.sum())
            if k < MIN_TRADES:
                continue
            base = getattr(self, f"_base_{period}_{name}")
            if not np.isfinite(base):
                continue
            e = float(ex[sel].mean() - base)
            edges.append(e)
            cls_pos[ASSET_CLASS.get(name, "crypto")].append(e > 0)
            trades += k
            n_ok += 1
        if n_ok < MIN_ASSETS:
            return dict(n_assets=n_ok, ok=False)
        edges = np.array(edges)
        return dict(n_assets=n_ok, ok=True, trades=trades,
                    edge_med=float(np.median(edges)),
                    edge_mean=float(edges.mean()),
                    edge_pos=float((edges > 0).mean()),
                    cls_min=min(np.mean(v) if v else 0 for v in cls_pos.values()))

    def full_score(self, atoms, trigger: str, period: str, rng) -> dict:
        """2단 — 순환이동 검정 포함한 정식 채점."""
        masks = self.is_mask if period == "IS" else self.oos_mask
        rows = []
        for name in self.data:
            ex = self.exits[name]
            sig = self.signal(name, atoms, trigger)
            sel = sig & masks[name] & ~np.isnan(ex)
            k = int(sel.sum())
            if k < MIN_TRADES:
                rows.append(dict(asset=name, trades=k))
                continue
            r = ex[sel]
            base = getattr(self, f"_base_{period}_{name}")
            rows.append(dict(
                asset=name, asset_class=ASSET_CLASS.get(name, "?"), trades=k,
                win_rate=float((r > 0).mean()), avg_return=float(r.mean()),
                pf=float(r[r > 0].sum() / -r[r <= 0].sum()) if (r <= 0).any() else np.inf,
                edge=float(r.mean() - base),
                p_shift=shift_test(sig, ex, masks[name], float(r.mean()), rng)))
        df = pd.DataFrame(rows)
        s = score(df)
        return s, df


def describe(atoms, trigger: str) -> str:
    parts = [f"{f}[t-{lag}]{op}{thr}" for f, lag, op, thr in atoms]
    if trigger != "none":
        parts.append(trigger)
    return " & ".join(parts)


def search(panel: SearchPanel, n_atoms: int = 2, top_k: int = 40) -> pd.DataFrame:
    print(f"1단 스크리닝: 조건 {n_atoms}개 조합 열거 중...")
    combos = list(itertools.combinations(range(len(ATOMS)), n_atoms))
    total = len(combos) * len(TRIGGERS)
    print(f"  후보 {total:,}개 (조건조합 {len(combos):,} × 방아쇠 {len(TRIGGERS)})")

    cands = []
    for ci, idx in enumerate(combos):
        atoms = [ATOMS[i] for i in idx]
        for trig in TRIGGERS:
            s = panel.quick_score(atoms, trig, "IS")
            if not s["ok"] or s["edge_pos"] < 0.65 or s["edge_med"] <= 0:
                continue
            cands.append(dict(atoms=atoms, trigger=trig, **s))
        if (ci + 1) % 200 == 0:
            print(f"  {ci+1:,}/{len(combos):,} 조합, 1단 통과 {len(cands):,}개")

    print(f"\n1단 통과: {len(cands):,}개 / {total:,}개")
    if not cands:
        return pd.DataFrame()

    df = pd.DataFrame(cands).sort_values(["edge_pos", "edge_med"], ascending=False)
    df = df.head(top_k).reset_index(drop=True)

    print(f"\n2단 정식 검정: 상위 {len(df)}개")
    rng = np.random.default_rng(SEED)
    out = []
    for i, r in df.iterrows():
        s_is, _ = panel.full_score(r.atoms, r.trigger, "IS", rng)
        s_oos, detail = panel.full_score(r.atoms, r.trigger, "OOS", rng)
        out.append(dict(desc=describe(r.atoms, r.trigger),
                        atoms=r.atoms, trigger=r.trigger,
                        **{f"IS_{k}": v for k, v in s_is.items()},
                        **{f"OOS_{k}": v for k, v in s_oos.items()}))
        print(f"  [{i+1:2d}/{len(df)}] {out[-1]['desc'][:58]:58s} "
              f"OOS edge+={s_oos.get('edge_pos', float('nan')):.2f} "
              f"p={s_oos.get('combined_p', float('nan')):.4f}")
    return pd.DataFrame(out)


if __name__ == "__main__":
    print("=" * 74)
    print("패널 기반 패턴 탐색")
    print("=" * 74)
    p = SearchPanel()
    res = search(p, n_atoms=2, top_k=40)
    if len(res):
        out = RESULTS_DIR / "pattern_search.csv"
        res.drop(columns=["atoms"]).to_csv(out, index=False)
        print(f"\n-> results/{out.name}")
        cols = ["desc", "IS_edge_pos", "IS_combined_p",
                "OOS_trades", "OOS_edge_med", "OOS_edge_pos",
                "OOS_pf_med", "OOS_combined_p"]
        print("\n" + res[[c for c in cols if c in res.columns]]
              .round(4).to_string(index=False))
