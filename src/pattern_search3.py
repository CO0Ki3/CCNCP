"""
조건 3개 패턴 전수 탐색 — C(156,3) × 방아쇠 5 = 3,103,100개

조건 2개 탐색(pattern_search.py)에서 Bonferroni 통과가 0건이었다.
탐색 공간이 좁았던 것인지, 아니면 애초에 그런 패턴이 없는 것인지를
가르려면 공간을 넓혀봐야 한다.

조건 2개 대비 달라지는 점
  1) 공간이 51배 — FlatPanel로 자산 루프를 없애고 단조 가지치기를 넣는다.
  2) 조건이 늘수록 신호가 희박해진다 — MIN_TRADES/MIN_ASSETS가 자연스러운
     정규화 역할을 하지만, 동시에 "표본이 적어 우연히 좋아 보이는" 후보가
     늘어난다. 그래서 2단 정식 검정 대상을 60개로 늘리고 Bonferroni도 그만큼
     엄격해진다 (0.05/60).
  3) 중복 후보 제거 — 조건이 3개면 사실상 같은 신호를 내는 조합이 많다
     (예: clspos≥0.7과 clspos≥0.5가 다른 조건에 의해 같은 날만 남는 경우).
     신호 마스크 해시로 중복을 걸러 2단 60칸을 서로 다른 형태로 채운다.

선택 편향에 대해
  IS에서 310만 개를 훑고 고른 최고 성적에는 반드시 요행이 섞인다.
  그러나 판정은 스크리닝에 쓰지 않은 OOS 패널에서, 실제로 수행한 검정 수(60)에
  대한 Bonferroni 문턱으로 내린다. 이 문턱은 스크리닝 후보 수와 무관하게 유효하다
  — 단, 같은 자산의 IS와 OOS가 완전히 독립은 아니라는 점은 남는 한계다.
"""
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from flat_panel import FlatPanel, viable
from panel_eval import MIN_ASSETS, MIN_TRADES, SEED
from pattern_search import ATOMS, SearchPanel, TRIGGERS, describe

RESULTS_DIR = Path(__file__).parent.parent / "results"
TOP_STRICT = 30      # 자산군 3개 모두에서 통하는 후보 (범용 요건)
TOP_OPEN = 30        # 자산군 제약 없이 성적만 좋은 후보 (대조군)

_FP = None           # 워커 프로세스별 전역


def _init(payload=None):
    """워커 초기화.

    macOS 파이썬은 multiprocessing 기본 시작 방식이 spawn이라 부모의 전역이
    워커로 상속되지 않는다. 그래서 귀무 대조처럼 변형된 패널을 쓰려면
    payload로 명시적으로 넘겨야 한다. payload가 없으면 워커가 직접 만든다.
    """
    global _FP
    if payload is not None:
        _FP = payload
    elif _FP is None:
        _FP = FlatPanel()


KEEP_PER_ATOM = 400      # 워커당 보관 상한 — 전량 보관하면 메모리가 터진다


def _scan_i(i: int):
    """원자 i를 첫 조건으로 하는 모든 (i<j<k) 조합을 훑는다.

    통과 후보를 전부 들고 있으면 메모리가 감당이 안 되므로, 워커 안에서
    (edge_pos, edge_med) 기준 상위 KEEP_PER_ATOM개만 남기고 통과 총수만 따로 센다.
    """
    fp = _FP
    A = len(ATOMS)
    ok = fp.ok["IS"]
    mi = fp.atoms[i] & ok
    if not viable(int(mi.sum())):
        return 0, []
    out, n_pass = [], 0
    for j in range(i + 1, A):
        mij = mi & fp.atoms[j]
        if not viable(int(mij.sum())):
            continue                     # 단조성: k를 붙여도 더 줄어들 뿐
        for k in range(j + 1, A):
            mijk = mij & fp.atoms[k]
            if not viable(int(mijk.sum())):
                continue
            for ti in range(len(fp.trig_names)):
                sel = mijk & fp.trig[ti]
                s = fp.score_mask(sel, "IS")
                if s is None or s["edge_pos"] < 0.65 or s["edge_med"] <= 0:
                    continue
                n_pass += 1
                out.append((i, j, k, ti, s["edge_pos"], s["edge_med"],
                            s["cls_min"], s["trades"], s["n_assets"],
                            hash(np.packbits(sel).tobytes())))
        if len(out) > KEEP_PER_ATOM * 4:
            out.sort(key=lambda r: (-r[4], -r[5]))
            del out[KEEP_PER_ATOM:]
    out.sort(key=lambda r: (-r[4], -r[5]))
    return n_pass, out[:KEEP_PER_ATOM]


def stage1(n_proc: int = 4, fp: "FlatPanel" = None) -> pd.DataFrame:
    total = 3103100
    print(f"1단 스크리닝 — C({len(ATOMS)},3) × 방아쇠 {len(TRIGGERS)} = "
          f"{total:,}개 후보")
    t0 = time.time()
    rows, n_pass = [], 0
    with Pool(n_proc, initializer=_init, initargs=(fp,)) as pool:
        for n, (cnt, part) in enumerate(
                pool.imap_unordered(_scan_i, range(len(ATOMS))), 1):
            rows += part
            n_pass += cnt
            if n % 20 == 0:
                print(f"  원자 {n}/{len(ATOMS)} 완료, 통과 {n_pass:,}개, "
                      f"{time.time()-t0:.0f}초", flush=True)
    print(f"1단 완료 {time.time()-t0:.0f}초 — 통과 {n_pass:,}개 / {total:,}개 "
          f"({n_pass/total:.1%}), 상위 {len(rows):,}개 보관")
    return pd.DataFrame(rows, columns=[
        "i", "j", "k", "ti", "edge_pos", "edge_med", "cls_min",
        "trades", "n_assets", "fp"])


def pick(df: pd.DataFrame) -> pd.DataFrame:
    """중복 신호를 걸러내고 엄격/개방 두 갈래로 상위 후보를 뽑는다."""
    df = df.sort_values(["edge_pos", "edge_med"], ascending=False)
    df = df.drop_duplicates(subset=["fp"])          # 같은 신호를 내는 조합 제거
    strict = df[df.cls_min >= 0.5].head(TOP_STRICT)
    rest = df[~df.index.isin(strict.index)].head(TOP_OPEN)
    strict = strict.assign(pool="strict")
    rest = rest.assign(pool="open")
    return pd.concat([strict, rest]).reset_index(drop=True)


def stage2(sp: SearchPanel, cand: pd.DataFrame) -> pd.DataFrame:
    print(f"\n2단 정식 검정 — {len(cand)}개 (순환이동 + 자산상관 보정)")
    rng = np.random.default_rng(SEED)
    trig_names = list(TRIGGERS)
    out = []
    for n, r in cand.iterrows():
        atoms = [ATOMS[int(r.i)], ATOMS[int(r.j)], ATOMS[int(r.k)]]
        trig = trig_names[int(r.ti)]
        s_is, _ = sp.full_score(atoms, trig, "IS", rng)
        s_oos, _ = sp.full_score(atoms, trig, "OOS", rng)
        out.append(dict(desc=describe(atoms, trig), pool=r.pool,
                        **{f"IS_{a}": b for a, b in s_is.items()},
                        **{f"OOS_{a}": b for a, b in s_oos.items()}))
        print(f"  [{n+1:2d}/{len(cand)}] {out[-1]['desc'][:50]:50s} "
              f"OOS edge+={s_oos.get('edge_pos', float('nan')):.2f} "
              f"n_eff={s_oos.get('n_eff', float('nan')):4.1f} "
              f"p={s_oos.get('combined_p', float('nan')):.4f}")
    return pd.DataFrame(out)


if __name__ == "__main__":
    print("=" * 78)
    print("조건 3개 패널 기반 패턴 탐색")
    print("=" * 78)
    raw = stage1()
    if raw.empty:
        print("1단 통과 후보 없음")
        sys.exit(0)
    cand = pick(raw)
    print(f"중복 제거 후 2단 대상: strict {int((cand.pool=='strict').sum())}개 / "
          f"open {int((cand.pool=='open').sum())}개")

    res = stage2(SearchPanel(), cand)
    res = res.sort_values("OOS_combined_p")
    res.to_csv(RESULTS_DIR / "pattern_search3.csv", index=False)

    cols = ["desc", "pool", "IS_edge_pos", "IS_combined_p", "OOS_trades",
            "OOS_edge_med", "OOS_edge_pos", "OOS_pf_med", "OOS_n_eff",
            "OOS_combined_p_naive", "OOS_combined_p"]
    print("\n" + res[[c for c in cols if c in res.columns]]
          .head(25).round(4).to_string(index=False))

    thr = 0.05 / len(res)
    n_pass = int((res.OOS_combined_p < thr).sum())
    n_nominal = int((res.OOS_combined_p < 0.05).sum())
    print(f"\n2단 {len(res)}개에 대한 Bonferroni 문턱 {thr:.6f} -> 통과 {n_pass}개")
    print(f"무보정 0.05 기준 {n_nominal}개 (우연 기대치 {0.05*len(res):.1f}개)")
    print(f"-> results/pattern_search3.csv")
