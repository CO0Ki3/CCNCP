"""
평평한 패널 — 30개 자산을 하나의 배열로 이어붙여 후보 채점을 벡터화한다.

왜 필요한가
  조건 2개 탐색은 후보 60,450개였고 자산마다 파이썬 루프를 돌아도 1.6분이면 끝났다.
  조건 3개는 C(156,3)×5 = 3,103,100개로 51배다. 자산당 루프를 유지하면 83분이 걸린다.

  자산별 배열을 하나로 이어붙이고 경계 색인을 들고 있으면,
  후보 하나의 채점이 "긴 배열 AND 3번 + 구간합 2번"으로 끝난다.
  파이썬 루프가 자산 수(30번)에서 0번으로 줄어 후보당 1.6ms → 0.1ms가 된다.

가지치기
  조건을 하나 더 AND하면 신호는 반드시 줄어든다(단조 감소). 따라서
  2개 조합 단계에서 이미 전체 신호가 하한(MIN_TRADES × MIN_ASSETS)에 못 미치면
  그 조합을 포함하는 모든 3개 조합도 볼 필요가 없다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import ASSET_CLASS
from panel_eval import MIN_ASSETS, MIN_TRADES, Panel
from pattern_search import ATOMS, TRIGGERS, atom_mask, features

CLS_IDX = {"crypto": 0, "equity": 1, "commodity_fx": 2}


class FlatPanel:
    """자산별 배열을 이어붙인 표현.

    구성 요소
      offsets   : 자산 경계 (np.add.reduceat 용)
      exits     : 이어붙인 거래 수익률 (진입 불가일은 0, valid로 구분)
      atoms     : (원자 수, 전체길이) bool — 조건 원자별 성립 여부
      trig      : (방아쇠 수, 전체길이) bool
      ok[period]: 구간 마스크 & 유효 진입일
      base[period]: 자산별 랜덤 진입 기준선
    """

    def __init__(self, panel: Panel = None):
        p = panel if panel is not None else Panel()
        self.panel = p
        self.names = list(p.data)
        lens = [len(p.data[n]) for n in self.names]
        self.offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
        self.total = int(sum(lens))
        self.cls = np.array([CLS_IDX[ASSET_CLASS.get(n, "crypto")]
                             for n in self.names])

        ex = [np.nan_to_num(p.exits[n], nan=0.0) for n in self.names]
        valid = [~np.isnan(p.exits[n]) for n in self.names]
        self.exits = np.concatenate(ex)

        feats = {n: features(p.data[n]) for n in self.names}
        self.atoms = np.zeros((len(ATOMS), self.total), dtype=bool)
        for ai, a in enumerate(ATOMS):
            self.atoms[ai] = np.concatenate([atom_mask(feats[n], a)
                                             for n in self.names])
        self.trig_names = list(TRIGGERS)
        self.trig = np.zeros((len(TRIGGERS), self.total), dtype=bool)
        for ti, t in enumerate(self.trig_names):
            self.trig[ti] = np.concatenate(
                [np.nan_to_num(TRIGGERS[t](p.data[n]), nan=0).astype(bool)
                 for n in self.names])

        self.ok, self.base = {}, {}
        for per, masks in (("IS", p.is_mask), ("OOS", p.oos_mask)):
            m = np.concatenate([masks[n] & v for n, v in zip(self.names, valid)])
            self.ok[per] = m
            cnt = np.add.reduceat(m, self.offsets)
            tot = np.add.reduceat(np.where(m, self.exits, 0.0), self.offsets)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.base[per] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    def __getstate__(self):
        """워커로 보낼 때 원본 Panel(30개 DataFrame)은 떼어낸다.

        _scan_i는 offsets·exits·atoms·trig·ok·base·cls만 쓴다. Panel까지
        같이 절이면 전송량이 수십 MB 늘어날 뿐 쓰이지 않는다.
        """
        st = self.__dict__.copy()
        st["panel"] = None
        return st

    # ── 채점 ──────────────────────────────────────
    def score_mask(self, sel: np.ndarray, period: str) -> dict:
        """이미 구간·유효성까지 AND된 신호 마스크를 자산별로 집계한다."""
        cnt = np.add.reduceat(sel, self.offsets)
        keep = cnt >= MIN_TRADES
        n_ok = int(keep.sum())
        if n_ok < MIN_ASSETS:
            return None
        tot = np.add.reduceat(np.where(sel, self.exits, 0.0), self.offsets)
        mean = tot[keep] / cnt[keep]
        edges = mean - self.base[period][keep]
        if not np.isfinite(edges).all():
            return None
        pos = edges > 0
        cls = self.cls[keep]
        cls_pos = [pos[cls == c].mean() if (cls == c).any() else 0.0
                   for c in range(3)]
        return dict(n_assets=n_ok, trades=int(cnt[keep].sum()),
                    edge_med=float(np.median(edges)),
                    edge_pos=float(pos.mean()),
                    cls_min=float(min(cls_pos)))

    def signal_mask(self, atom_idx, trig_idx: int, period: str) -> np.ndarray:
        m = self.trig[trig_idx] & self.ok[period]
        for ai in atom_idx:
            m = m & self.atoms[ai]
        return m


def viable(total_hits: int) -> bool:
    """필요조건: 전 자산 합계 신호가 최소 표본 요구량에 못 미치면 볼 필요 없다.

    자산 MIN_ASSETS개에서 각각 MIN_TRADES건이 나오려면 합계가 최소 그 곱이어야 한다.
    조건을 더해도 신호는 늘지 않으므로, 여기서 걸리면 하위 조합 전부 탈락이다.
    """
    return total_hits >= MIN_TRADES * MIN_ASSETS
