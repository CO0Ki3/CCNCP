"""
탐색 결과 보정 검정 — "최고 성적"이 신호인지 탐색 규모의 부산물인지 가른다.

문제
  310만 개를 훑어 고른 최고 후보의 OOS p가 0.05라면, 그것은 좋은 것인가?
  단일 검정이었다면 유의하다. 그러나 우리는 60번 검정했고, 그 60개는
  310만 개 중에서 IS 성적으로 골라낸 것이다.

두 가지를 본다
  1) Bonferroni — 실제 수행한 60번의 검정에 대한 문턱 0.05/60.
     스크리닝 후보 수와 무관하게 유효하다(OOS는 스크리닝에 쓰지 않았으므로).
  2) p값 분포 자체 — 전역 귀무가설이 참이면 60개의 OOS p는 균등분포여야 한다.
     탐색이 실제 신호를 건졌다면 분포가 0쪽으로 쏠린다.
     KS 검정과 "0.05 미만 개수 vs 기대치 3개"로 확인한다.

  2번이 중요한 이유: Bonferroni는 개별 후보에 대한 판정이라 보수적이다.
  개별로는 아무도 문턱을 못 넘어도 60개 전체가 균등분포에서 유의하게
  벗어나 있다면 "약하지만 실재하는 신호가 여러 후보에 퍼져 있다"는 뜻이 된다.
  반대로 균등하다면 탐색이 건진 것은 잡음뿐이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kstest

RESULTS_DIR = Path(__file__).parent.parent / "results"


def calibrate(csv: str, label: str) -> None:
    path = RESULTS_DIR / csv
    if not path.exists():
        print(f"[skip] {csv} 없음")
        return
    res = pd.read_csv(path)
    p = res["OOS_combined_p"].dropna().to_numpy()
    n = len(p)
    if n == 0:
        print(f"[skip] {csv}: OOS p값 없음")
        return

    thr = 0.05 / n
    n_bonf = int((p < thr).sum())
    n_nom = int((p < 0.05).sum())
    exp_nom = 0.05 * n
    ks = kstest(p, "uniform")
    bt = binomtest(n_nom, n, 0.05, alternative="greater")

    print(f"\n{'='*72}\n{label}  (2단 검정 {n}개)\n{'='*72}")
    print(f"  최소 OOS p          : {p.min():.4f}")
    print(f"  Bonferroni 문턱     : {thr:.6f}  -> 통과 {n_bonf}개")
    print(f"  무보정 0.05 미만    : {n_nom}개 (우연 기대치 {exp_nom:.1f}개), "
          f"이항검정 p={bt.pvalue:.3f}")
    print(f"  p값 균등성 KS 검정  : D={ks.statistic:.3f}, p={ks.pvalue:.3f}")
    print(f"  p값 사분위          : {np.percentile(p,[25,50,75]).round(3)}")

    if n_bonf > 0:
        verdict = "합격 후보 있음 — 개별 확인 필요"
    elif bt.pvalue < 0.05 or ks.pvalue < 0.05:
        verdict = ("개별 합격은 없으나 p값 분포가 균등에서 유의하게 벗어남 "
                   "— 약한 신호가 퍼져 있을 가능성")
    else:
        verdict = "합격 없음. p값 분포도 균등과 구분되지 않음 — 잡음"
    print(f"  판정: {verdict}")

    if "OOS_combined_p_naive" in res:
        nv = res["OOS_combined_p_naive"].dropna()
        print(f"  참고: 자산상관 보정 전 0.05 미만은 {(nv<0.05).sum()}개였다 "
              f"(보정 후 {n_nom}개)")


if __name__ == "__main__":
    calibrate("pattern_search.csv", "조건 2개 탐색 (후보 60,450개)")
    calibrate("pattern_search3.csv", "조건 3개 탐색 (후보 3,103,100개)")
