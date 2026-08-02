"""
다중자산 일봉 OHLCV 로더 (범용성 검증용)

출처
  - 암호화폐: Bitfinex public API (BTC 원본 데이터와 동일 거래소 → 형식 일관성)
  - 주식/ETF/원자재/통화: Nasdaq public API

과적합 방지 원칙 4-2-3(시장 보편성) 검증을 위해, BTC 외 자산에서도
동일한 패턴 정의·동일한 백테스트 파라미터로 성과를 확인한다.

한 번 받은 데이터는 data/ 에 CSV로 캐싱하여 재실행 시 네트워크 의존을 없앤다.
"""
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36")

# ──────────────────────────────────────────────
# 검증 대상 자산
# 자산군을 나눈 이유: 같은 자산군 내 상관은 높으므로,
# "암호화폐에서만 되는 패턴"인지 "시장 구조 전반에 통하는 패턴"인지 구분하려면
# 자산군별로 성과를 따로 집계해야 한다.
# ──────────────────────────────────────────────
CRYPTO = {
    "ETH": "tETHUSD", "LTC": "tLTCUSD", "XRP": "tXRPUSD", "EOS": "tEOSUSD",
    "XMR": "tXMRUSD", "ZEC": "tZECUSD", "DASH": "tDSHUSD", "IOTA": "tIOTUSD",
    "NEO": "tNEOUSD", "ETC": "tETCUSD", "SOL": "tSOLUSD", "ADA": "tADAUSD",
    "DOT": "tDOTUSD", "LINK": "tLINK:USD",
}

# (심볼, nasdaq assetclass)
EQUITY = {
    "SPY": ("SPY", "etf"), "QQQ": ("QQQ", "etf"), "IWM": ("IWM", "etf"),
    "EEM": ("EEM", "etf"), "XLE": ("XLE", "etf"),
    "AAPL": ("AAPL", "stocks"), "MSFT": ("MSFT", "stocks"),
    "NVDA": ("NVDA", "stocks"), "JPM": ("JPM", "stocks"),
}
COMMODITY_FX = {
    "GLD": ("GLD", "etf"), "SLV": ("SLV", "etf"), "USO": ("USO", "etf"),
    "FXE": ("FXE", "etf"), "FXY": ("FXY", "etf"), "UUP": ("UUP", "etf"),
}

ASSET_CLASS = dict(BTC="crypto")
ASSET_CLASS.update({k: "crypto" for k in CRYPTO})
ASSET_CLASS.update({k: "equity" for k in EQUITY})
ASSET_CLASS.update({k: "commodity_fx" for k in COMMODITY_FX})


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.csv"


def _normalize(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """index=DatetimeIndex, columns=[open,high,low,close,volume] 로 통일."""
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # 0 또는 음수 가격은 데이터 오류 → 제거 (비율 기반 패턴이 발산함)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]

    # 봉 무결성 복구.
    # Nasdaq 원본에 high < open 인 행이 존재한다(2023-06-05, IWM/JPM/XLE 각 1행).
    # 이 프로젝트의 모든 패턴은 꼬리·몸통 비율로 정의되므로 high < open 이면
    # 위꼬리가 음수가 되어 패턴 판정이 조용히 오염된다. 행을 삭제하면 t-1/t-2
    # 참조가 하루를 건너뛰므로, 삭제 대신 high/low를 시가·종가를 감싸도록 넓힌다.
    hi = df[["high", "open", "close"]].max(axis=1)
    lo = df[["low", "open", "close"]].min(axis=1)
    n_fixed = int(((hi != df["high"]) | (lo != df["low"])).sum())
    if n_fixed:
        # 이 보정은 OHLC 불변식을 데이터 품질 게이트로 못 쓰게 만든다.
        # 컬럼 순서를 잘못 매핑하거나 API 스키마가 바뀌어도 조용히 통과하게 되므로,
        # 산발적 오타(0.1% 이하)만 보정하고 그 이상이면 실패시킨다.
        rate = n_fixed / len(df)
        if rate > 0.001:
            raise ValueError(
                f"{name or '?'}: 봉 무결성 위반 {n_fixed}/{len(df)}행 ({rate:.2%}). "
                f"산발적 오타 수준을 넘었다 — 컬럼 매핑이나 API 스키마를 확인할 것.")
        print(f"  [fix]  {name or '?'}: 봉 무결성 위반 {n_fixed}행 보정 ({rate:.3%})")
    df["high"], df["low"] = hi, lo
    return df


def _fetch_bitfinex(symbol: str) -> pd.DataFrame:
    """Bitfinex는 요청당 최대 10,000봉. 일봉 기준 전체 이력을 한 번에 커버한다."""
    r = requests.get(
        f"https://api-pub.bitfinex.com/v2/candles/trade:1D:{symbol}/hist",
        params={"limit": 10000, "sort": 1},
        headers={"User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"빈 응답: {symbol}")
    # Bitfinex 캔들 순서: [mts, open, close, high, low, volume]
    df = pd.DataFrame(rows, columns=["mts", "open", "close", "high", "low", "volume"])
    df["time"] = pd.to_datetime(df["mts"], unit="ms")
    return _normalize(df.set_index("time"), symbol)


def _fetch_nasdaq(symbol: str, assetclass: str) -> pd.DataFrame:
    r = requests.get(
        f"https://api.nasdaq.com/api/quote/{symbol}/historical",
        params={"assetclass": assetclass, "fromdate": "2010-01-01",
                "todate": "2026-12-31", "limit": 100000},
        headers={"User-Agent": UA},
        timeout=40,
    )
    r.raise_for_status()
    rows = r.json()["data"]["tradesTable"]["rows"]
    if not rows:
        raise ValueError(f"빈 응답: {symbol}")
    df = pd.DataFrame(rows).rename(columns={"close": "close_last"})

    def num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str).str.replace(r"[$,]", "", regex=True).replace({"N/A": None}),
            errors="coerce",
        )

    out = pd.DataFrame({
        "open": num(df["open"]), "high": num(df["high"]),
        "low": num(df["low"]), "close": num(df["close_last"]),
        "volume": num(df["volume"]),
    })
    out.index = pd.to_datetime(df["date"], format="%m/%d/%Y")
    out.index.name = "time"
    return _normalize(out.dropna(subset=["open", "high", "low", "close"]), symbol)


def load_asset(name: str, refresh: bool = False) -> pd.DataFrame:
    """단일 자산 로드. 캐시 우선."""
    path = _cache_path(name)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _normalize(df, name)

    if name in CRYPTO:
        df = _fetch_bitfinex(CRYPTO[name])
    elif name in EQUITY:
        df = _fetch_nasdaq(*EQUITY[name])
    elif name in COMMODITY_FX:
        df = _fetch_nasdaq(*COMMODITY_FX[name])
    else:
        raise KeyError(f"알 수 없는 자산: {name}")

    df.to_csv(path)
    return df


def load_btc() -> pd.DataFrame:
    """프로젝트 원본 BTC 데이터 (기준 자산)."""
    from eda import load_data
    return _normalize(load_data(), "BTC")


def load_all(refresh: bool = False, min_rows: int = 800) -> dict:
    """전 자산 로드. 실패하거나 표본이 부족한 자산은 건너뛴다.

    min_rows=800: 롤링 window 60일을 소진하고도 IS/OOS 양쪽에서
    최소한의 신호 표본이 남으려면 3년 이상의 이력이 필요하다.
    """
    out = {"BTC": load_btc()}
    for name in list(CRYPTO) + list(EQUITY) + list(COMMODITY_FX):
        try:
            df = load_asset(name, refresh=refresh)
            if len(df) < min_rows:
                print(f"  [skip] {name}: 이력 부족 ({len(df)}일 < {min_rows})")
                continue
            out[name] = df
            print(f"  [ok]   {name:5s} {len(df):5d}일  "
                  f"{df.index[0].date()} ~ {df.index[-1].date()}")
        except Exception as e:
            print(f"  [fail] {name}: {type(e).__name__} {str(e)[:60]}")
        if not _cache_path(name).exists():
            time.sleep(0.3)   # 신규 수집 시에만 API 예의상 간격
    return out


if __name__ == "__main__":
    print("=" * 64)
    print("다중자산 데이터 수집")
    print("=" * 64)
    data = load_all()
    print(f"\n총 {len(data)}개 자산 확보")
    for cls in ["crypto", "equity", "commodity_fx"]:
        names = [n for n in data if ASSET_CLASS.get(n, "crypto") == cls]
        print(f"  {cls:13s}: {len(names)}개  {', '.join(names)}")
