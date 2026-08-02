"""
범용성 검증 결과 시각화 — 자산별 "랜덤 진입 대비 초과수익(edge)".

edge를 그리는 이유: PF나 총수익을 그리면 자산이 오른 만큼이 그대로 막대가 되어
패턴의 기여를 볼 수 없다. edge는 같은 구간·같은 청산규칙으로 아무 날에나
진입했을 때의 평균수익을 빼낸 값이라, 0을 기준으로 패턴의 정보량만 남는다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
POS = "#2a78d6"      # 발산 팔레트 한랭극
NEG = "#e34948"      # 발산 팔레트 온난극
GRID = "#e6e5e1"


def plot(strategy: str = "v2_combo", fname: str = "generality_edge.png"):
    res = pd.read_csv(RESULTS_DIR / "multi_asset_validation.csv")
    g = (res[(res.period == "OOS") & (res.strategy == strategy) & (res.trades >= 10)]
         .dropna(subset=["edge"]).sort_values("edge"))

    edge = g.edge.to_numpy() * 100
    names = g.asset.tolist()
    colors = [POS if e > 0 else NEG for e in edge]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(range(len(edge)), edge, color=colors, height=0.62)

    ax.axvline(0, color=INK, linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color=INK2)
    ax.set_xlabel("랜덤 진입 대비 거래당 초과수익 (%p)", fontsize=10, color=INK2)

    # BTC만 직접 라벨 — 패턴을 설계한 자산이라 대조군 역할을 한다
    if "BTC" in names:
        i = names.index("BTC")
        ax.text(edge[i] + 0.12, i, f"BTC  +{edge[i]:.2f}%p (패턴 설계 자산)",
                va="center", fontsize=9, color=INK, fontweight="bold")
        ax.get_yticklabels()[i].set_color(INK)
        ax.get_yticklabels()[i].set_fontweight("bold")

    n_pos = int((edge > 0).sum())
    ax.set_title(
        f"v2 전략은 BTC 밖에서 재현되지 않는다\n"
        f"30개 자산 OOS · 초과수익 양수 {n_pos}/{len(edge)}개 — 동전던지기와 구분 불가",
        fontsize=13, color=INK, loc="left", pad=14)

    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.margins(y=0.01)
    # BTC 직접 라벨이 축 밖으로 밀리지 않도록 오른쪽 여유를 준다
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.42)

    fig.tight_layout()
    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> results/{fname}  (양수 {n_pos}/{len(edge)})")


if __name__ == "__main__":
    import matplotlib
    for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"]:
        if any(cand in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    plot()
