"""
다중자산 범용성 검증 (과적합 방지 원칙 4-2-3)

목적
  회의 #009 미결 안건 — CloseDominance(strict) 편입 여부를 다중자산으로 판정하고,
  나아가 v2 확정 패턴 3종이 BTC 전용인지 시장 구조 전반의 패턴인지 판별한다.

핵심 설계: 랜덤 진입 대비 초과 성과(edge)
  Long only 전략은 상승 자산에서 "아무 날에나" 진입해도 수익이 난다.
  따라서 PF 1.8 같은 절대 수치만으로는 패턴에 edge가 있는지 알 수 없다.
  같은 구간·같은 청산 규칙·같은 거래 수로 무작위 진입을 2,000회 부트스트랩하여
  패턴 평균수익이 그 분포의 몇 백분위에 있는지(=경험적 p-value)를 측정한다.
  이것이 "패턴이 정보를 담고 있는가"에 대한 직접적인 검정이다.

백테스트 규칙은 strategy.py(v2)와 동일: 신호 다음날 시가 진입,
트레일링 스탑 10%, 최대 보유 10일, 왕복 수수료 0.2%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import ASSET_CLASS, load_all
from pattern_research import pattern_close_dominance
from patterns import (pattern_bear_absorption, pattern_tail_echo,
                      pattern_triple_rise)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TEST_START = "2021-01-01"
TRAIL_PCT = 0.10
MAX_HOLD = 10
FEE = 0.001
N_BOOT = 2000
SEED = 20260802     # 고정 시드 — 재현성 확보 (결과에 맞춰 시드를 고르는 행위는 금지)


# ──────────────────────────────────────────────
# 진입일별 청산 결과 사전 계산
# strategy.py::backtest_trailing 과 동일한 규칙을 모든 날짜에 대해 한 번만 계산해 둔다.
# 이렇게 하면 "i일에 신호가 났다면 얼마를 벌었나"가 O(1) 조회가 되어
# 부트스트랩 2,000회 × 30자산 × 6전략이 현실적인 시간에 끝난다.
# ──────────────────────────────────────────────
def precompute_exits(df: pd.DataFrame, trail_pct=TRAIL_PCT,
                     max_hold=MAX_HOLD, fee=FEE, realistic=False,
                     with_index=False):
    """returns[i] = i일 종가에 신호가 났을 때의 거래 수익률 (불가능하면 NaN).

    realistic=False가 기본값인 이유는 v2 원본 규칙(strategy.py)을 그대로
    재현하기 위해서다. 원본에는 두 가지 낙관 편향이 있다.

      1) 루프가 d=1부터라 진입 봉의 고가·저가가 무시된다. 트레일링 스탑이
         max(진입가, 이후 고가)가 아니라 진입가에 고정 앵커되어 느슨해진다.
      2) 스탑 발동 시 exit_price=stop 으로 스탑 가격 정확 체결을 가정한다.
         실제로는 발동일 시가가 이미 스탑 아래인 경우가 BTC 기준 21.8%,
         그때 평균 -3.14% 아래에서 체결된다. 이 편향은 갭이 큰 암호화폐에
         선택적으로 몰려서(crypto 17.0% vs equity 0.9%) 자산군 비교를 오염시킨다.

    realistic=True는 두 편향을 모두 제거한다: 진입 봉부터 감시하고,
    체결가를 min(stop, 발동일 시가)로 잡는다.
    """
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    out = np.full(n, np.nan)
    xidx = np.full(n, -1, dtype=int)
    start = 0 if realistic else 1

    for i in range(n):
        entry_i = i + 1
        if entry_i >= n:
            continue
        entry_price = o[entry_i]
        peak = entry_price
        exit_price = None
        exit_i = -1

        for d in range(start, max_hold + 1):
            day_i = entry_i + d
            if day_i >= n:
                exit_price, exit_i = c[day_i - 1], day_i - 1
                break
            peak = max(peak, h[day_i])
            stop = peak * (1 - trail_pct)
            if l[day_i] <= stop:
                # 갭하락이면 스탑가에 못 붙는다 — 시가가 이미 아래면 시가 체결
                exit_price = min(stop, o[day_i]) if realistic else stop
                exit_i = day_i
                break
            if d == max_hold:
                exit_price, exit_i = c[day_i], day_i

        if exit_price is not None:
            out[i] = (exit_price / entry_price - 1) - 2 * fee
            xidx[i] = exit_i
    return (out, xidx) if with_index else out


def daily_curve(df: pd.DataFrame, sig_sel: np.ndarray, rets: np.ndarray,
                xidx: np.ndarray, max_positions: int = 1) -> np.ndarray:
    """일별 자산곡선. 자본을 max_positions 등분해 슬롯이 빌 때만 진입한다.

    거래 순서로 나열해 복리하는 방식은 두 가지를 숨긴다.
      - 동시에 열린 포지션이 함께 무너지는 구간이 평활화되어 MDD가 얕게 나온다
        (실측: 175개 케이스 전부에서 시간축 MDD가 더 깊었고 중앙 3.2%p 차이).
      - 거래가 한 건도 없는 달이 집계에서 사라진 채 연환산된다
        (BTC OOS 65개월 중 15개월이 거래 0건).
    자본 제약과 시간축을 모두 넣어야 실현 가능한 성과가 된다.
    """
    n = len(df)
    equity = np.ones(n)
    slot_free_at = np.zeros(max_positions, dtype=int)   # 각 슬롯이 비는 날
    w = 1.0 / max_positions
    cash = 1.0
    pend = []                                           # (청산일, 손익기여)

    for i in np.where(sig_sel)[0]:
        entry_i = i + 1
        if entry_i >= n or xidx[i] < 0:
            continue
        free = np.where(slot_free_at <= entry_i)[0]
        if len(free) == 0:
            continue                                    # 슬롯 만석 → 신호 포기
        slot_free_at[free[0]] = xidx[i] + 1
        pend.append((xidx[i], rets[i]))

    pend.sort()
    j = 0
    for t in range(n):
        while j < len(pend) and pend[j][0] == t:
            cash *= (1 + w * pend[j][1])
            j += 1
        equity[t] = cash
    return equity


def curve_metrics(equity: np.ndarray, days: int, bars_per_year: float) -> dict:
    peak = np.maximum.accumulate(equity)
    mdd = float(((equity - peak) / peak).min())
    years = days / bars_per_year
    cagr = float(equity[-1] ** (1 / years) - 1) if years > 0 and equity[-1] > 0 else np.nan
    return dict(cagr=cagr, mdd_time=mdd, total=float(equity[-1] - 1))


BARS_PER_YEAR = {"crypto": 365.0, "equity": 252.0, "commodity_fx": 252.0}


def metrics(rets: np.ndarray, years: float = None) -> dict:
    """거래 순서 기준 지표.

    Sharpe 연율화 계수를 sqrt(365/MAX_HOLD)=6.04로 두면
    "연 36.5건을 쉼 없이 이어 붙인다"를 가정하게 된다. 실제 거래 빈도는
    패널 중앙 9.5건/년이라 이 계수는 Sharpe를 2.2배 부풀린다. 게다가
    주식·ETF는 연 252봉이므로 365 가정이 암호화폐 대비 1.2배를 더 얹는다.
    그래서 years가 주어지면 실제 거래빈도 sqrt(n/years)로 연율화한다.
    """
    n = len(rets)
    if n == 0:
        return dict(trades=0, win_rate=np.nan, avg_return=np.nan,
                    sharpe=np.nan, mdd=np.nan, pf=np.nan)
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    mdd = float(((cum - peak) / peak).min())
    sd = rets.std(ddof=1) if n > 1 else 0.0
    factor = np.sqrt(n / years) if years and years > 0 else np.sqrt(365 / MAX_HOLD)
    sharpe = float(rets.mean() / sd * factor) if sd > 0 else np.nan
    gains, losses = rets[rets > 0], rets[rets <= 0]
    pf = float(gains.sum() / -losses.sum()) if losses.sum() < 0 else np.inf
    return dict(trades=n, win_rate=float((rets > 0).mean()),
                avg_return=float(rets.mean()), sharpe=sharpe, mdd=mdd, pf=pf)


def random_edge(exits: np.ndarray, mask: np.ndarray, n_trades: int,
                observed_mean: float, rng: np.random.Generator) -> dict:
    """같은 구간에서 무작위 진입 n_trades건을 N_BOOT회 반복한 분포 대비 위치.

    p_value = 무작위 진입이 패턴 평균수익 이상을 낸 비율.
    작을수록 패턴에 정보가 있다는 뜻.
    """
    pool = exits[mask & ~np.isnan(exits)]
    if len(pool) < 30 or n_trades == 0 or not np.isfinite(observed_mean):
        return dict(p_value=np.nan, rand_mean=np.nan, edge=np.nan)
    draws = rng.choice(pool, size=(N_BOOT, n_trades), replace=True).mean(axis=1)
    return dict(p_value=float((draws >= observed_mean).mean()),
                rand_mean=float(pool.mean()),
                edge=float(observed_mean - pool.mean()))


def shift_test(sig: np.ndarray, exits: np.ndarray, mask: np.ndarray,
               observed_mean: float, rng: np.random.Generator) -> float:
    """신호 계열 순환이동 검정 — random_edge보다 보수적이고 정확하다.

    패턴 신호는 특정 국면에 몰려서 발생한다(군집성). 그런데 random_edge의
    iid 부트스트랩은 진입일들이 서로 독립이라고 가정하므로, 실제 유효표본이
    거래 수보다 적은데도 분산을 과소평가해 p-value를 낙관적으로 만든다.

    대신 신호 계열을 무작위 lag만큼 원형으로 돌린다. 이러면 신호의
    발생 간격·군집 구조는 그대로 보존한 채 가격과의 정렬만 깨진다.
    귀무가설 "신호 타이밍은 가격과 무관하다"에 대한 직접 검정이 된다.

    이동은 반드시 평가 구간 안에서만 이루어져야 한다. 계열 전체를 굴린 뒤
    구간 마스크와 교집합하면 구간 밖 신호가 안으로 흘러들어와 귀무분포의
    거래 수가 관측 거래 수와 달라진다(실측: 관측 57건인데 귀무 중앙 67건).
    그러면 "같은 거래 수로 무작위 진입했을 때"라는 비교 조건이 깨진다.
    """
    idx = np.where(mask & ~np.isnan(exits))[0]
    m = len(idx)
    if not np.isfinite(observed_mean) or m == 0:
        return np.nan
    inwin = sig[idx]                      # 구간 내부만 잘라낸 신호 계열
    k = int(inwin.sum())
    if k == 0:
        return np.nan
    vals = exits[idx]
    lags = rng.integers(1, m, size=N_BOOT)
    draws = np.array([vals[np.roll(inwin, int(lag))].mean() for lag in lags])
    return float((draws >= observed_mean).mean())


# ──────────────────────────────────────────────
# 검증 대상 전략
# ──────────────────────────────────────────────
def _combine(df, fns):
    combo = pd.Series("none", index=df.index)
    for fn in fns:
        s = fn(df)
        combo[(combo == "none") & (s == "long")] = "long"
    return combo


V2_PARTS = [pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise]

STRATEGIES = {
    "TailEcho":        pattern_tail_echo,
    "BearAbsorption":  pattern_bear_absorption,
    "TripleRise":      pattern_triple_rise,
    "CloseDominance":  pattern_close_dominance,
    "v2_combo":        lambda df: _combine(df, V2_PARTS),
    "v2+CloseDom":     lambda df: _combine(df, V2_PARTS + [pattern_close_dominance]),
}


def evaluate_asset(name: str, df: pd.DataFrame, rng: np.random.Generator,
                   realistic: bool = False) -> list:
    exits, xidx = precompute_exits(df, realistic=realistic, with_index=True)
    bpy = BARS_PER_YEAR[ASSET_CLASS.get(name, "crypto")]
    dates = df.index
    is_oos = dates >= pd.Timestamp(TEST_START)
    # 롤링 window 60일 + 참조 3일이 채워지기 전 구간은 신호가 성립할 수 없으므로 제외.
    # 무작위 진입 풀도 같은 구간으로 맞춰야 비교가 공정하다.
    warm = np.zeros(len(df), dtype=bool)
    warm[63:] = True
    is_train = warm & ~is_oos
    is_test = warm & is_oos

    rows = []
    for sname, fn in STRATEGIES.items():
        sig = (fn(df) == "long").to_numpy()
        for period, mask in (("IS", is_train), ("OOS", is_test)):
            sel = sig & mask & ~np.isnan(exits)
            rets = exits[sel]
            years = int(mask.sum()) / bpy
            m = metrics(rets, years)
            e = random_edge(exits, mask, m["trades"], m["avg_return"], rng)
            p_shift = shift_test(sig, exits, mask, m["avg_return"], rng)
            # 자본 1구좌 제약(중복 진입 금지)을 건 시간축 자산곡선
            eq = daily_curve(df, sel, exits, xidx, max_positions=1)
            eq = eq[mask] / eq[mask][0] if mask.any() else np.ones(1)
            cm = curve_metrics(eq, int(mask.sum()), bpy)
            # 벤치마크: 같은 구간 단순 보유
            px = df["close"].to_numpy()[mask]
            bh = float(px[-1] / px[0] - 1) if len(px) > 1 else np.nan
            rows.append(dict(asset=name, asset_class=ASSET_CLASS.get(name, "?"),
                             strategy=sname, period=period,
                             days=int(mask.sum()), **m, **e, p_shift=p_shift,
                             **cm, buyhold=bh, beats_bh=bool(cm["total"] > bh)))
    return rows


def run():
    print("=" * 74)
    print("다중자산 범용성 검증 — v2 패턴 + CloseDominance(strict)")
    print(f"trail={TRAIL_PCT:.0%}, max_hold={MAX_HOLD}일, fee={FEE:.1%}(편도), "
          f"부트스트랩 {N_BOOT}회")
    print("=" * 74)

    data = load_all()
    rng = np.random.default_rng(SEED)
    rows = []
    for name, df in data.items():
        rows += evaluate_asset(name, df, rng)
        print(f"  평가 완료: {name}")

    res = pd.DataFrame(rows)
    out = RESULTS_DIR / "multi_asset_validation.csv"
    res.to_csv(out, index=False)
    print(f"\n-> {out.relative_to(Path(__file__).parent.parent)}")
    return res


def _bh_survivors(pvals, alpha: float = 0.05) -> int:
    """Benjamini-Hochberg 절차로 FDR alpha에서 살아남는 검정 수.

    자산 29개 × 전략 6종 = 174번 검정하면 무보정 5% 기준으로는
    평균 8.7건이 그냥 우연히 유의하게 나온다. 보정 없이 "p<0.05가 N개"를
    세는 것은 성과가 아니라 검정 횟수를 세는 것에 가깝다.
    """
    p = np.sort(np.asarray(pvals, dtype=float))
    p = p[~np.isnan(p)]
    m = len(p)
    if m == 0:
        return 0
    below = np.where(p <= alpha * np.arange(1, m + 1) / m)[0]
    return int(below[-1] + 1) if len(below) else 0


def summarize(res: pd.DataFrame) -> None:
    oos = res[res.period == "OOS"]

    print("\n" + "=" * 74)
    print("[1] 전략별 OOS 종합 (BTC 제외 — 29개 자산)")
    print("=" * 74)
    ex = oos[oos.asset != "BTC"]
    print(f"{'전략':16s} {'자산':>4s} {'거래계':>6s} {'PF중앙':>7s} {'Sharpe중앙':>10s} "
          f"{'edge>0':>7s} {'p_sh<.05':>9s} {'FDR생존':>8s} {'보유압도':>9s}")
    for s in STRATEGIES:
        g = ex[(ex.strategy == s) & (ex.trades >= 10)]
        if g.empty:
            print(f"{s:16s} {'-':>4s} (표본 부족)")
            continue
        print(f"{s:16s} {len(g):4d} {int(g.trades.sum()):6d} "
              f"{g.pf.median():7.2f} {g.sharpe.median():10.2f} "
              f"{(g.edge > 0).mean():6.0%} {(g.p_shift < 0.05).mean():8.0%} "
              f"{_bh_survivors(g.p_shift):7d}건 {g.beats_bh.mean():8.0%}")
    print("  * 기준선: 29개 자산에서 우연히 p<0.05가 나오는 기대 비율이 5%다. "
          "관측치가 5% 근처면 edge 없음.")
    print("  * FDR생존: Benjamini-Hochberg 5%로 다중검정 보정 후 살아남은 자산 수.")
    print("  * 보유압도: 같은 구간 단순 보유(buy&hold)를 이긴 자산 비율.")

    print("\n" + "=" * 74)
    print("[2] 자산군별 OOS PF 중앙값")
    print("=" * 74)
    piv = (oos[oos.trades >= 10]
           .pivot_table(index="strategy", columns="asset_class", values="pf",
                        aggfunc="median"))
    print(piv.round(2).to_string())

    print("\n" + "=" * 74)
    print("[3] 랜덤 진입 대비 초과수익(edge) 중앙값, %p")
    print("=" * 74)
    piv2 = (oos[oos.trades >= 10]
            .pivot_table(index="strategy", columns="asset_class", values="edge",
                         aggfunc="median") * 100)
    print(piv2.round(3).to_string())

    print("\n" + "=" * 74)
    print("[4] BTC 대조 (원본 확정 근거)")
    print("=" * 74)
    btc = res[(res.asset == "BTC") & (res.period == "OOS")]
    print(btc[["strategy", "trades", "win_rate", "avg_return", "sharpe",
               "mdd", "mdd_time", "pf", "cagr", "buyhold", "edge",
               "p_shift"]].round(4).to_string(index=False))
    print("  mdd = 거래순서 자산곡선 / mdd_time = 일별 시간축 (중복진입 금지, 자본 1구좌)")
    print("  cagr = 시간축 실현 복리수익률 / buyhold = 같은 구간 단순 보유")
    print("\n  * 다중검정 보정: 프로젝트가 시험한 패턴은 최소 10종이므로,")
    print("    BTC 단일 자산 p-value는 Bonferroni 기준 0.05/10 = 0.005 이하라야 유의하다.")

    print("\n" + "=" * 74)
    print("[5] IS→OOS 성과 유지율 (BTC 제외, PF 중앙값)")
    print("=" * 74)
    piv3 = (res[(res.trades >= 10) & (res.asset != "BTC")]
            .pivot_table(index="strategy", columns="period", values="pf",
                         aggfunc="median"))
    piv3["OOS/IS"] = (piv3["OOS"] / piv3["IS"]).round(2)
    print(piv3.round(2).to_string())


if __name__ == "__main__":
    result = run()
    summarize(result)
