"""
경험적 귀무 대조 — 이 탐색 파이프라인이 "신호가 전혀 없을 때" 무엇을 내놓는지 잰다.

왜 필요한가
  조건 2개 탐색의 2단 40개에서, 자산상관 보정 전에는 p<0.05가 10개(우연 기대 2개),
  보정 후에는 0개(기대 2개)였다. 전자는 지나치게 관대하고 후자는 지나치게
  엄격해 보인다. 어느 쪽이 맞는지 이론으로 따지는 대신 직접 재면 된다.

방법
  가격과 신호의 정렬만 깨고 나머지는 전부 보존한다. 각 자산의 청산결과 배열
  exits를 자기 길이의 일정 비율만큼 순환이동시킨다. 이러면
    - 캔들 특징량, 조건 원자, 방아쇠, 신호의 군집 구조: 그대로
    - 수익률의 분포·자기상관·자산 간 상관: 그대로
    - "이 신호가 이 수익을 낳았다"는 대응 관계: 파괴
  이 상태로 동일한 탐색을 돌렸을 때 나오는 최소 p값이 곧
  "탐색 규모만으로 얻어지는 최고 성적"이다.

해석
  실제 탐색의 최소 p가 귀무 대조의 최소 p보다 뚜렷이 작아야 신호라 할 수 있다.
  비슷하면, 그 후보는 310만 번 뽑기의 당첨자일 뿐이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from flat_panel import FlatPanel
from panel_eval import SEED
from pattern_search import SearchPanel
import pattern_search3 as ps3

RESULTS_DIR = Path(__file__).parent.parent / "results"


def roll_exits(panel, frac: float = 0.37) -> None:
    """자산별 exits를 순환이동해 신호-수익 대응을 파괴한다 (제자리 수정).

    frac은 자산 길이에 대한 비율. 0.37처럼 어중간한 값을 쓰는 이유는
    계절성이나 반감기 주기와 우연히 맞아떨어지지 않게 하기 위함이다.
    이동 후 SearchPanel이 미리 계산해 둔 랜덤 진입 기준선도 다시 계산한다.
    """
    for name, ex in panel.exits.items():
        lag = int(len(ex) * frac)
        panel.exits[name] = np.roll(ex, lag)
    if hasattr(panel, "trig"):        # SearchPanel의 구간별 기준선 갱신
        for name in panel.data:
            ex = panel.exits[name]
            for per, mask in (("IS", panel.is_mask[name]),
                              ("OOS", panel.oos_mask[name])):
                pool = ex[mask & ~np.isnan(ex)]
                setattr(panel, f"_base_{per}_{name}",
                        float(pool.mean()) if len(pool) else np.nan)


def run_null(frac: float = 0.37, out: str = None) -> pd.DataFrame:
    # 이동폭마다 다른 파일명으로 저장한다. 고정 파일명을 쓰면 반복 실행이
    # 서로를 덮어써서, 나중에 반복 간 비교를 하려 할 때 표본이 소실된다.
    out = out or f"null_search3_{frac}.csv"
    print("=" * 78)
    print(f"경험적 귀무 대조 — exits를 자산 길이의 {frac:.0%}만큼 순환이동")
    print("=" * 78)

    sp = SearchPanel()
    roll_exits(sp, frac)

    # spawn 방식에서는 부모 전역이 워커로 안 넘어가므로 명시적으로 전달한다
    fp = FlatPanel(sp)
    raw = ps3.stage1(fp=fp)
    if raw.empty:
        print("귀무 대조에서 1단 통과 후보 없음 — 탐색이 잡음을 전혀 못 건짐")
        return pd.DataFrame()
    cand = ps3.pick(raw)
    res = ps3.stage2(sp, cand).sort_values("OOS_combined_p")
    res.to_csv(RESULTS_DIR / out, index=False)

    p = res.OOS_combined_p.dropna().to_numpy()
    print(f"\n[귀무 대조 결과] 2단 {len(p)}개")
    print(f"  최소 p        : {p.min():.4f}")
    print(f"  p<0.05 개수   : {int((p<0.05).sum())} (기대 {0.05*len(p):.1f})")
    print(f"  p 사분위      : {np.percentile(p,[25,50,75]).round(3)}")
    print(f"-> results/{out}")
    return res


if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.37
    run_null(frac)
