"""
패널 평가기 — 임의의 후보 패턴을 30개 자산 전체에서 한 번에 채점한다.

왜 필요한가
  기존 프로젝트는 BTC 한 종목에서 패턴을 고르고 BTC OOS로 검증했다.
  이 방식은 "BTC 2013~2020에 맞는 형태"를 고르게 되어 있고, 실제로
  multi_asset_validation.py 결과 v2 패턴 3종은 다른 29개 자산에서
  랜덤 진입 대비 우위가 없었다.

  선택 자체를 패널에서 해야 한다. 30개 자산 IS 구간에서 일관된 우위를 보이는
  패턴만 후보로 남기고, 그 다음 OOS 패널에서 확인한다. 자산 30개는 사실상
  30번의 독립 재현 시도이므로, 여기서 살아남으면 BTC 우연일 확률이 매우 낮다.

선택 기준(IS 패널)과 확인 기준(OOS 패널)을 분리해서 쓰는 것이 이 파일의 요점이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import ASSET_CLASS, load_all
from multi_asset_validation import (MAX_HOLD, TEST_START, metrics,
                                    precompute_exits, random_edge, shift_test)

MIN_TRADES = 8         # 자산당 최소 거래 수 — 이하면 그 자산은 집계에서 제외
MIN_ASSETS = 15        # 최소 이 정도 자산에서 신호가 나야 "범용"이라 말할 수 있음
# 자산당 8건은 그 자산 하나만 보면 무의미한 표본이다. 다만 여기서는 자산별
# 판정이 목적이 아니라 자산 15~30개의 약한 증거를 Stouffer로 합산하는 것이
# 목적이므로, 저빈도 패턴을 평가 대상에서 통째로 탈락시키는 쪽이 더 해롭다.
SEED = 20260802


class Panel:
    """자산별 가격·청산결과·구간마스크를 한 번만 계산해 재사용한다.

    후보 패턴을 수십~수백 개 돌릴 것이므로, 자산마다 매번 exits를 다시
    계산하면(자산당 ~4,800일 × 10일 루프) 탐색이 불가능해진다.
    """

    def __init__(self, data: dict = None, split: str = "median"):
        """split
          "fixed"  — 2021-01-01 기준 (기존 프로젝트 관례, 결과 비교용)
          "median" — 자산별 자기 이력의 중앙 시점 기준 (패턴 탐색용, 기본값)

        median을 기본으로 두는 이유가 둘 있다.
        첫째, 고정 분할은 Nasdaq 자산의 IS가 2016~2020 4년뿐이고 SOL·ADA·DOT·LINK는
        IS가 아예 없어서, 저빈도 패턴은 IS 패널이 6~8개 자산으로 쪼그라든다.
        둘째, 고정 분할이면 30개 자산의 OOS가 전부 같은 2021~2026 국면이라
        검정 30개가 사실상 1개나 마찬가지다. 자산마다 OOS 시기를 흩뜨리면
        "그 시기에 우연히 통했다"가 반복될 확률이 급격히 낮아진다.
        """
        self.data = data if data is not None else load_all()
        self.split = split
        self.exits, self.is_mask, self.oos_mask = {}, {}, {}
        for name, df in self.data.items():
            self.exits[name] = precompute_exits(df)
            if split == "fixed":
                oos = np.asarray(df.index >= pd.Timestamp(TEST_START))
            else:
                oos = np.arange(len(df)) >= len(df) // 2
            warm = np.zeros(len(df), dtype=bool)
            warm[63:] = True
            # OOS 시작 직후 63일도 워밍업이 IS 구간을 참조하므로 배제할 필요는 없다
            # (참조 대상이 과거일 뿐 미래가 아니다). warm은 계열 시작부만 잘라낸다.
            self.is_mask[name] = warm & ~oos
            self.oos_mask[name] = warm & oos

    def evaluate(self, fn, period: str = "OOS", rng=None) -> pd.DataFrame:
        """후보 패턴 fn(df)->Series를 전 자산에서 평가. 자산당 1행."""
        rng = rng or np.random.default_rng(SEED)
        masks = self.oos_mask if period == "OOS" else self.is_mask
        rows = []
        for name, df in self.data.items():
            try:
                sig = (fn(df) == "long").to_numpy()
            except Exception as e:
                rows.append(dict(asset=name, trades=0, error=type(e).__name__))
                continue
            ex, mask = self.exits[name], masks[name]
            sel = sig & mask & ~np.isnan(ex)
            m = metrics(ex[sel])
            if m["trades"] < MIN_TRADES:
                rows.append(dict(asset=name,
                                 asset_class=ASSET_CLASS.get(name, "?"), **m))
                continue
            e = random_edge(ex, mask, m["trades"], m["avg_return"], rng)
            rows.append(dict(asset=name, asset_class=ASSET_CLASS.get(name, "?"),
                             **m, **e,
                             p_shift=shift_test(sig, ex, mask, m["avg_return"], rng)))
        return pd.DataFrame(rows)


def score(panel_df: pd.DataFrame) -> dict:
    """패널 결과 한 장을 요약 점수로 압축한다.

    핵심은 pf가 아니라 edge다. 상승 자산에서는 랜덤 진입도 pf>1이 나오므로
    pf는 자산의 방향성을 재는 것이지 패턴의 정보량을 재는 것이 아니다.

    combined_p: 자산별 p_shift를 Stouffer 방식으로 합산.
      개별 자산은 표본이 작아 유의하지 않아도, 같은 방향의 약한 우위가
      여러 자산에 걸쳐 반복되면 전체로는 유의해질 수 있다. 반대로 한두 자산의
      큰 성과가 나머지의 무성과에 희석되므로 요행에 강하다.
    """
    g = panel_df[panel_df.trades >= MIN_TRADES].dropna(subset=["edge"])
    n = len(g)
    if n < MIN_ASSETS:
        return dict(n_assets=n, ok=False, reason=f"자산 {n}개 < {MIN_ASSETS}")

    from scipy.stats import norm
    z = norm.isf(g.p_shift.clip(1e-4, 1 - 1e-4)).sum() / np.sqrt(n)
    return dict(
        n_assets=n, ok=True,
        trades=int(g.trades.sum()),
        trades_per_asset=float(g.trades.median()),
        edge_med=float(g.edge.median()),
        edge_pos=float((g.edge > 0).mean()),
        pf_med=float(g.pf.median()),
        win_med=float(g.win_rate.median()),
        sig_frac=float((g.p_shift < 0.05).mean()),
        combined_z=float(z),
        combined_p=float(norm.sf(z)),
        # 자산군 편중 확인 — 한 자산군에서만 되는 패턴은 범용이 아니다
        edge_pos_crypto=float((g[g.asset_class == "crypto"].edge > 0).mean()),
        edge_pos_equity=float((g[g.asset_class == "equity"].edge > 0).mean()),
        edge_pos_cfx=float((g[g.asset_class == "commodity_fx"].edge > 0).mean()),
    )


def passes(s: dict) -> bool:
    """범용 패턴 합격 기준.

    단일 자산 백테스트 지표(PF>1.5 등)를 쓰지 않는 이유:
    그 기준은 자산이 오르기만 해도 통과한다. 대신 "랜덤 진입 대비 우위가
    자산 전반에 걸쳐 반복되는가"만 본다.
    """
    return (s.get("ok")
            and s["edge_pos"] >= 0.65        # 자산의 2/3 이상에서 랜덤 대비 우위
            and s["combined_p"] < 0.01       # 합산 유의성
            and s["edge_med"] > 0
            and min(s["edge_pos_crypto"], s["edge_pos_equity"],
                    s["edge_pos_cfx"]) >= 0.5)   # 특정 자산군 전용이 아닐 것


def report(name: str, panel: Panel, fn, rng=None) -> dict:
    """IS로 고르고 OOS로 확인하는 2단 평가."""
    rng = rng or np.random.default_rng(SEED)
    s_is = score(panel.evaluate(fn, "IS", rng))
    s_oos = score(panel.evaluate(fn, "OOS", rng))
    return dict(pattern=name,
                **{f"IS_{k}": v for k, v in s_is.items()},
                **{f"OOS_{k}": v for k, v in s_oos.items()},
                IS_pass=passes(s_is), OOS_pass=passes(s_oos))


def fmt(rows: list) -> pd.DataFrame:
    cols = ["pattern", "IS_n_assets", "IS_trades", "IS_edge_med", "IS_edge_pos",
            "IS_combined_p", "IS_pass",
            "OOS_trades", "OOS_edge_med", "OOS_edge_pos", "OOS_pf_med",
            "OOS_combined_p", "OOS_pass"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


if __name__ == "__main__":
    from pattern_research import pattern_close_dominance
    from patterns import (pattern_bear_absorption, pattern_tail_echo,
                          pattern_triple_rise)

    print("패널 구성 중...")
    p = Panel()
    rng = np.random.default_rng(SEED)
    rows = [report(n, p, f, rng) for n, f in [
        ("TailEcho", pattern_tail_echo),
        ("BearAbsorption", pattern_bear_absorption),
        ("TripleRise", pattern_triple_rise),
        ("CloseDominance", pattern_close_dominance),
    ]]
    print("\n기존 확정 패턴의 패널 성적")
    print(fmt(rows).round(4).to_string(index=False))
