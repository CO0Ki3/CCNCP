"""
보유기간에 따른 edge 스케일링 — "시간봉으로 가면 되는가"에 답한다.

문제
  회의 #011에서 병목이 검정력으로 밝혀졌다. 자산당 거래 수를 20배로 늘리면
  최소 탐지 가능 효과크기(MDE)가 √20 ≈ 4.5배 개선되므로, 일봉을 시간봉으로
  바꾸면 통계적 격차는 메울 수 있다.

  그런데 통계만 보고 결정하면 안 된다. 보유기간이 짧아지면 거래당 수익의
  절대 크기도 함께 줄어드는데, **거래비용은 줄지 않기** 때문이다.
  왕복 0.2%는 10일 보유든 10시간 보유든 똑같이 나간다.

방법
  일봉 데이터 안에서 max_hold를 1~20일로 바꿔가며 같은 패턴의
  거래당 edge(랜덤 진입 대비)를 측정한다. edge가 √보유기간에 비례한다면,
  보유기간을 1/24로 줄일 때 edge는 1/4.9로 줄어든다. 그 값이 왕복 수수료
  아래로 내려가면 시간봉 경로는 통계적으로만 유리하고 경제적으로는 죽는다.

이 파일의 결론이 프로젝트 종료 여부를 가른다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import ASSET_CLASS
from multi_asset_validation import FEE, TRAIL_PCT, precompute_exits
from panel_eval import MIN_TRADES, Panel
from pattern_search import ATOMS, SearchPanel, describe

HOLDS = [1, 2, 3, 5, 10, 20]

# 회의 #011 최상위 후보 + v2 조합 구성 패턴
TOP = ([("lower", 3, "<=", 0.5), ("gap", 1, ">=", 0.0), ("gap", 2, "<=", -0.2)],
       "close<prev_low")


def edge_by_hold(sp: SearchPanel, atoms, trigger: str) -> pd.DataFrame:
    """보유기간별 거래당 edge 중앙값 (전 자산, OOS)."""
    rows = []
    for hold in HOLDS:
        edges, grosses, ns = [], [], 0
        for name, df in sp.data.items():
            ex = precompute_exits(df, trail_pct=TRAIL_PCT, max_hold=hold, fee=FEE)
            mask = sp.oos_mask[name]
            sel = sp.signal(name, atoms, trigger) & mask & ~np.isnan(ex)
            if sel.sum() < MIN_TRADES:
                continue
            pool = ex[mask & ~np.isnan(ex)]
            e = float(ex[sel].mean() - pool.mean())
            edges.append(e)
            grosses.append(float(np.abs(ex[sel]).mean()))
            ns += int(sel.sum())
        if len(edges) < 10:
            continue
        rows.append(dict(hold=hold, assets=len(edges), trades=ns,
                         edge_med=float(np.median(edges)),
                         edge_pos=float(np.mean(np.array(edges) > 0)),
                         abs_ret_med=float(np.median(grosses))))
    return pd.DataFrame(rows)


def report(tab: pd.DataFrame) -> None:
    base = tab[tab.hold == 10].iloc[0]
    print(f"\n{'보유':>4s} {'자산':>4s} {'거래':>6s} {'edge중앙':>9s} "
          f"{'edge>0':>7s} {'|수익|중앙':>10s} {'edge/√hold':>11s}")
    for _, r in tab.iterrows():
        print(f"{int(r.hold):4d} {int(r.assets):4d} {int(r.trades):6d} "
              f"{r.edge_med:+9.4%} {r.edge_pos:7.0%} {r.abs_ret_med:10.2%} "
              f"{r.edge_med/np.sqrt(r.hold):+11.4%}")

    print("\n[스케일링 진단]")
    sub = tab[tab.hold.isin([1, 2, 3, 5, 10, 20])]
    if len(sub) >= 3 and (sub.edge_med > 0).all():
        b = np.polyfit(np.log(sub.hold), np.log(sub.edge_med), 1)[0]
        print(f"  edge ∝ hold^{b:.2f}  (√스케일이면 0.50, 선형이면 1.00)")
    else:
        print("  일부 보유기간에서 edge가 음수라 로그 회귀 불가 — 표로 직접 판단")

    print("\n[시간봉 전환 시 손익분기]")
    print(f"  왕복 수수료 {2*FEE:.2%} (슬리피지 미포함)")
    for _, r in tab.iterrows():
        # 일봉 hold일 -> 시간봉 같은 봉수면 실제 경과시간이 1/24
        scaled = r.edge_med / np.sqrt(24)
        verdict = "생존" if scaled > 2 * FEE else "적자"
        print(f"  일봉 {int(r.hold):2d}일 edge {r.edge_med:+.3%} "
              f"-> 시간봉 환산 {scaled:+.3%}  vs 비용 {2*FEE:.2%}  [{verdict}]")


if __name__ == "__main__":
    sp = SearchPanel()
    atoms, trig = TOP
    print("=" * 72)
    print(f"보유기간별 edge 스케일링 — {describe(atoms, trig)}")
    print("=" * 72)
    tab = edge_by_hold(sp, atoms, trig)
    report(tab)
    tab.to_csv(Path(__file__).parent.parent / "results" / "holding_scaling.csv",
               index=False)
