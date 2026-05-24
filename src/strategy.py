"""
NCP 최종 전략: TailEcho + BearAbsorption 조합 (Long only)
- TailEcho    : 전일 위꼬리 P80 초과 후 오늘 고점 돌파 → Long
- BearAbsorption: 강한 하락을 강한 상승이 완전 회복 → Long
- 손절: 진입가 -7%
- 보유: 3일 후 청산
- 신호 중복 시 TailEcho 우선
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from eda import load_data
from patterns import pattern_tail_echo, pattern_bear_absorption
from backtest import backtest, compute_metrics, equity_curve, plot_equity, walk_forward

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TEST_START = "2021-01-01"


def combined_signals(df: pd.DataFrame) -> pd.Series:
    sig_te = pattern_tail_echo(df)
    sig_ba = pattern_bear_absorption(df)
    combo  = sig_te.copy()
    combo[(combo == "none") & (sig_ba == "long")] = "long"
    return combo


def run():
    print("=" * 60)
    print("NCP 최종 전략 실행")
    print("=" * 60)

    df = load_data()

    # ── 전체 구간 ──
    signals = combined_signals(df)
    trades  = backtest(df, signals, direction="long", stop_loss=-0.07)
    oos     = trades[trades["entry_date"] >= TEST_START].copy()

    te_mask = trades["entry_date"].isin(
        backtest(df, pattern_tail_echo(df), direction="long")["entry_date"]
    )
    ba_count = (~te_mask).sum()

    print(f"\n전체 거래: {len(trades)}건 ({len(trades)/13:.1f}건/년)")
    print(f"  TailEcho    : {te_mask.sum()}건")
    print(f"  BearAbsorption: {ba_count}건")
    print(f"\n검증 구간 (2021~2026):")
    m = compute_metrics(oos, "Combo OOS")
    for k, v in m.items():
        print(f"  {k}: {v}")

    # ── Walk-forward ──
    print("\nWalk-forward:")
    wf = walk_forward(df, combined_signals, stop_loss=-0.07, direction="long")
    m_wf = compute_metrics(wf, "WF")
    for k, v in m_wf.items():
        print(f"  {k}: {v}")

    # ── 차트 ──
    c1 = equity_curve(trades[trades["entry_date"] < TEST_START].copy(), "Train")
    c2 = equity_curve(oos, "Test (OOS)")
    c3 = equity_curve(wf, "Walk-forward")
    plot_equity([c1, c2], "NCP Final — Train vs Test", "final_train_test.png")
    plot_equity([c3],     "NCP Final — Walk-forward",  "final_wf.png")

    print("\n[완료] 최종 전략 실행 종료")
    print("=" * 60)
    return trades, oos, wf


if __name__ == "__main__":
    run()
