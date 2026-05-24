import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "BITFINEX_BTCUSD, 1D.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    stats = {
        "기간": f"{df.index[0].date()} ~ {df.index[-1].date()}",
        "총 거래일": len(df),
        "일 수익률 평균": f"{df['return'].mean()*100:.3f}%",
        "일 수익률 표준편차": f"{df['return'].std()*100:.3f}%",
        "왜도(skewness)": f"{df['return'].skew():.3f}",
        "첨도(kurtosis)": f"{df['return'].kurt():.3f}",
        "상승일 비율": f"{(df['return'] > 0).mean()*100:.1f}%",
        "최대 일 수익": f"{df['return'].max()*100:.1f}%",
        "최대 일 손실": f"{df['return'].min()*100:.1f}%",
        "평균 몸통 비율": f"{df['body_ratio'].mean()*100:.1f}%",
    }
    return stats, df


def classify_regime(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    시장 국면 분류 (MA50 기반 + 변동성)
    - Bull: 종가 > MA50 & 변동성 낮음
    - Bear: 종가 < MA50 & 변동성 낮음
    - Volatile_Bull: 종가 > MA50 & 변동성 높음
    - Volatile_Bear: 종가 < MA50 & 변동성 높음
    """
    df = df.copy()
    df["ma50"] = df["close"].rolling(window).mean()
    df["vol20"] = df["log_return"].rolling(20).std()
    vol_median = df["vol20"].median()

    conditions = [
        (df["close"] >= df["ma50"]) & (df["vol20"] <= vol_median),
        (df["close"] >= df["ma50"]) & (df["vol20"] > vol_median),
        (df["close"] < df["ma50"]) & (df["vol20"] <= vol_median),
        (df["close"] < df["ma50"]) & (df["vol20"] > vol_median),
    ]
    labels = ["Bull", "Volatile_Bull", "Bear", "Volatile_Bear"]
    df["regime"] = np.select(conditions, labels, default="Unknown")
    return df


def plot_price_with_regime(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    colors = {"Bull": "#2ecc71", "Volatile_Bull": "#f39c12",
              "Bear": "#e74c3c", "Volatile_Bear": "#9b59b6", "Unknown": "#95a5a6"}

    ax = axes[0]
    ax.semilogy(df.index, df["close"], color="#2c3e50", linewidth=0.8)
    for regime, color in colors.items():
        mask = df["regime"] == regime
        ax.fill_between(df.index, df["close"].min(), df["close"],
                        where=mask, alpha=0.15, color=color, label=regime)
    ax.set_ylabel("Price (log scale)")
    ax.set_title("BTCUSD 1D — Market Regime Classification")
    ax.legend(loc="upper left", fontsize=8)

    axes[1].bar(df.index, df["volume"], color="#3498db", alpha=0.5, width=1)
    axes[1].set_ylabel("Volume")

    axes[2].plot(df.index, df["return"] * 100, color="#7f8c8d", linewidth=0.5)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("Daily Return (%)")

    fig.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_overview.png", dpi=150)
    plt.close()
    print("  -> results/eda_overview.png 저장 완료")


def plot_return_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["return"].dropna() * 100, bins=100, color="#3498db",
                 edgecolor="white", linewidth=0.3)
    axes[0].set_title("Daily Return Distribution")
    axes[0].set_xlabel("Return (%)")
    axes[0].set_ylabel("Frequency")

    from scipy import stats as scipy_stats
    (mu, sigma) = scipy_stats.norm.fit(df["log_return"].dropna())
    x = np.linspace(df["log_return"].min(), df["log_return"].max(), 200)
    axes[1].hist(df["log_return"].dropna(), bins=100, density=True,
                 color="#2ecc71", edgecolor="white", linewidth=0.3, alpha=0.7)
    axes[1].plot(x, scipy_stats.norm.pdf(x, mu, sigma), "r--", linewidth=1.5, label="Normal fit")
    axes[1].set_title("Log Return Distribution vs Normal")
    axes[1].set_xlabel("Log Return")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_return_dist.png", dpi=150)
    plt.close()
    print("  -> results/eda_return_dist.png 저장 완료")


def regime_summary(df: pd.DataFrame):
    summary = df.groupby("regime").agg(
        일수=("close", "count"),
        평균수익률=("return", lambda x: f"{x.mean()*100:.3f}%"),
        승률=("return", lambda x: f"{(x > 0).mean()*100:.1f}%"),
        평균변동성=("vol20", lambda x: f"{x.mean()*100:.2f}%"),
    )
    return summary


def check_missing(df: pd.DataFrame):
    gaps = df.index.to_series().diff().dt.days
    large_gaps = gaps[gaps > 2]
    return large_gaps


def run():
    print("=" * 50)
    print("Phase 1: EDA 시작")
    print("=" * 50)

    print("\n[1] 데이터 로드...")
    df = load_data()
    print(f"  로드 완료: {len(df)}행")

    print("\n[2] 기초 통계...")
    stats, df = basic_stats(df)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n[3] 결측/갭 확인...")
    gaps = check_missing(df)
    print(f"  2일 이상 갭: {len(gaps)}건")
    if len(gaps) > 0:
        print(f"  최대 갭: {gaps.max():.0f}일 ({gaps.idxmax().date()})")

    print("\n[4] 시장 국면 분류...")
    df = classify_regime(df)
    regime_dist = df["regime"].value_counts()
    print(regime_dist.to_string())

    print("\n[5] 국면별 통계...")
    print(regime_summary(df).to_string())

    print("\n[6] 차트 생성...")
    plot_price_with_regime(df)
    plot_return_distribution(df)

    print("\n[완료] EDA 종료")
    print("=" * 50)
    return df


if __name__ == "__main__":
    df = run()
