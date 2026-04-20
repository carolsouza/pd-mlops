"""
eda.py — Análise Exploratória para o dataset Telco Customer Churn.

Execução: python eda/eda.py  (à partir da raiz do projeto mlops-churn)

Saídas:
    outputs/eda/stats/   — CSVs e JSONs com as estatísticas
    outputs/eda/figures/ — PNGs referenciados no preprocessing.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "processed" / "telco_customer_churn.parquet"
STATS_DIR   = BASE_DIR / "outputs" / "eda" / "stats"
FIGURES_DIR = BASE_DIR / "outputs" / "eda" / "figures"

STATS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI    = 120
STYLE  = "seaborn-v0_8-whitegrid"
TARGET = "Churn"

try:
    plt.style.use(STYLE)
except Exception:
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"  [fig] {path.name}")


def _save_json(data: dict, name: str) -> None:
    def _convert(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, (np.bool_,)):     return bool(obj)
        raise TypeError(type(obj))

    path = STATS_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_convert)
    print(f"  [json] {path.name}")


def _save_csv(df: pd.DataFrame, name: str) -> None:
    path = STATS_DIR / name
    df.to_csv(path)
    print(f"  [csv] {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 1 — Carregamento e visão geral
# ═══════════════════════════════════════════════════════════════════════════════

def secao_1_visao_geral(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Carrega o parquet, documenta dtypes e identifica o problema de TotalCharges.
    Retorna o DataFrame com TotalCharges convertido para float.
    """
    print("\n── Seção 1 · Visão geral e tipo de TotalCharges ──────────────────")

    info = {
        "rows":      int(df_raw.shape[0]),
        "columns":   int(df_raw.shape[1]),
        "dtypes":    {col: str(dtype) for col, dtype in df_raw.dtypes.items()},
        "TotalCharges_dtype_original": str(df_raw["TotalCharges"].dtype),
        "TotalCharges_nao_numericos": int(
            pd.to_numeric(df_raw["TotalCharges"], errors="coerce").isna().sum()
        ),
    }
    _save_json(info, "01_visao_geral.json")
    print(f"  Shape: {df_raw.shape}")
    print(f"  TotalCharges dtype original: {info['TotalCharges_dtype_original']}")
    print(f"  TotalCharges não numéricos : {info['TotalCharges_nao_numericos']} registros")
    print("  preprocessing.yaml: type_cast TotalCharges -> float64")

    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 2 — Valores ausentes e flag is_new_customer
# ═══════════════════════════════════════════════════════════════════════════════

def secao_2_ausentes_e_flag(df: pd.DataFrame) -> None:
    print("\n── Seção 2 · Valores ausentes e raiz dos NaN ────────────────────")

    nan_rows = df[df["TotalCharges"].isna()][["tenure", "MonthlyCharges", "TotalCharges"]]
    print(f"  Linhas com TotalCharges=NaN: {len(nan_rows)}")
    print(f"  tenure dessas linhas -> todos zero: {(nan_rows['tenure'] == 0).all()}")
    print(f"  MonthlyCharges dessas linhas:\n{nan_rows['MonthlyCharges'].describe().round(2)}")
    print("  -> preprocessing.yaml: imputation mediana + binary_flag is_new_customer (tenure=0)")

    missing = {
        "TotalCharges_nan_count": int(df["TotalCharges"].isna().sum()),
        "tenure_zero_count":      int((df["tenure"] == 0).sum()),
        "todos_nan_sao_tenure0":  bool((nan_rows["tenure"] == 0).all()),
    }
    _save_json(missing, "02_missing_values.json")

    # Figura: distribuição de tenure nos NaN vs resto
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["tenure"].hist(bins=40, ax=axes[0], color="steelblue", alpha=0.8)
    axes[0].axvline(0, color="crimson", linestyle="--", linewidth=1.8, label="tenure=0")
    axes[0].set_title("Distribuição de tenure (dataset completo)")
    axes[0].set_xlabel("tenure (meses)")
    axes[0].legend()

    tenure_zero = df[df["tenure"] == 0]["TotalCharges"].isna().value_counts()
    axes[1].bar(["tenure=0\nTotalCharges=NaN", "tenure=0\nTotalCharges OK"],
                [tenure_zero.get(True, 0), tenure_zero.get(False, 0)],
                color=["crimson", "steelblue"], alpha=0.8)
    axes[1].set_title("tenure=0: todos os 11 NaN de TotalCharges")
    axes[1].set_ylabel("Contagem")

    fig.suptitle("Seção 2 — Raiz dos NaN: clientes no 1º mês (tenure=0)", fontsize=12)
    _save_fig(fig, "fig_02_missing_tenure_zero.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 3 — Distribuição do target (desbalanceamento)
# ═══════════════════════════════════════════════════════════════════════════════

def secao_3_target(df: pd.DataFrame) -> None:
    print("\n── Seção 3 · Distribuição do target Churn ────────────────────────")

    counts = df[TARGET].value_counts()
    pcts   = df[TARGET].value_counts(normalize=True).round(4)
    print(f"  No (0): {counts[0]} ({pcts[0]*100:.1f}%)")
    print(f"  Yes(1): {counts[1]} ({pcts[1]*100:.1f}%)")
    print("  -> preprocessing.yaml: target nota 73.5%/26.5% (desbalanceado)")

    _save_json({
        "churn_no":  int(counts[0]),
        "churn_yes": int(counts[1]),
        "pct_no":    float(pcts[0]),
        "pct_yes":   float(pcts[1]),
    }, "03_target_distribution.json")

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Não cancela (0)", "Cancela (1)"], [counts[0], counts[1]],
                  color=["steelblue", "tomato"], alpha=0.85)
    for bar, pct in zip(bars, [pcts[0], pcts[1]]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{pct*100:.1f}%", ha="center", fontsize=11)
    ax.set_title("Distribuição do Target — Churn\n(dataset desbalanceado)", fontsize=12)
    ax.set_ylabel("Contagem de clientes")
    _save_fig(fig, "fig_03_target_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 4 — Distribuição e skewness das numéricas
# ═══════════════════════════════════════════════════════════════════════════════

def secao_4_distribuicoes_numericas(df: pd.DataFrame) -> None:
    print("\n── Seção 4 · Distribuições numéricas e skewness ─────────────────")

    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    skews = df[num_cols].skew().round(4)
    print(f"  Skewness:\n{skews.to_string()}")
    print("  -> log_transform aplicado em TotalCharges (skew=0.96) e tenure")
    _save_csv(skews.to_frame("skewness"), "04_skewness.csv")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        # original
        axes[0, i].hist(data, bins=40, color="steelblue", alpha=0.8, density=True)
        axes[0, i].set_title(f"{col}\nskew={data.skew():.2f}", fontsize=9)
        # log1p (só para TotalCharges e tenure; outros mantém original)
        if col in ("TotalCharges", "tenure"):
            log_data = np.log1p(data)
            axes[1, i].hist(log_data, bins=40, color="seagreen", alpha=0.8, density=True)
            axes[1, i].set_title(f"log1p({col})\nskew={log_data.skew():.2f}", fontsize=9)
        else:
            axes[1, i].hist(data, bins=40, color="lightgray", alpha=0.6, density=True)
            axes[1, i].set_title(f"{col} (sem log)\nskew={data.skew():.2f}", fontsize=9)

    axes[0, 0].set_ylabel("Densidade (original)", fontsize=9)
    axes[1, 0].set_ylabel("Densidade (pós log1p)", fontsize=9)
    fig.suptitle("Seção 4 — Distribuições Numéricas: antes e depois de log1p", fontsize=12)
    _save_fig(fig, "fig_04_distribuicoes_numericas.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 5 — Churn rate por Contract
# ═══════════════════════════════════════════════════════════════════════════════

def secao_5_churn_por_contrato(df: pd.DataFrame) -> None:
    print("\n── Seção 5 · Churn rate por tipo de contrato ────────────────────")

    churn_rate = (
        df.groupby("Contract")[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
        .sort_values("churn_rate", ascending=False)
    )
    churn_rate["churn_rate"] = churn_rate["churn_rate"].round(4)
    print(churn_rate.to_string())
    print("  -> Contract: ordinal Month-to-month=0, One year=1, Two year=2")
    _save_csv(churn_rate, "05_churn_por_contrato.csv")

    ORDER = ["Month-to-month", "One year", "Two year"]
    fig, ax = plt.subplots(figsize=(7, 4))
    rates = [churn_rate.loc[c, "churn_rate"] for c in ORDER]
    bars = ax.bar(ORDER, rates, color=["tomato", "orange", "steelblue"], alpha=0.85)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{rate*100:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Taxa de Churn")
    ax.set_ylim(0, 0.60)
    ax.set_title("Seção 5 — Churn rate por Tipo de Contrato\n"
                 "(justifica encoding ordinal: Month-to-month=0 -> Two year=2)", fontsize=11)
    _save_fig(fig, "fig_05_churn_por_contrato.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 6 — Churn rate por InternetService e colunas "No service"
# ═══════════════════════════════════════════════════════════════════════════════

def secao_6_internet_e_ternarios(df: pd.DataFrame) -> None:
    print("\n── Seção 6 · InternetService e encoding ternário ────────────────")

    # InternetService
    churn_internet = (
        df.groupby("InternetService")[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
        .sort_values("churn_rate", ascending=False)
    )
    print("  Churn rate por InternetService:")
    print(churn_internet.round(4).to_string())

    # Colunas ternárias: diferença de churn entre "No service", "No" e "Yes"
    ternary_cols = ["OnlineSecurity", "TechSupport", "StreamingTV"]
    records = []
    for col in ternary_cols:
        for val in ["No internet service", "No", "Yes"]:
            mask = df[col] == val
            if mask.sum() == 0:
                continue
            records.append({
                "feature":   col,
                "value":     val,
                "churn_rate": df.loc[mask, TARGET].mean().round(4),
                "n":         int(mask.sum()),
            })
    tern_df = pd.DataFrame(records)
    print("\n  Churn rate nas colunas ternárias (amostra):")
    print(tern_df.to_string(index=False))
    print("  -> No internet service=0, No=1, Yes=2 (hierarquia preservada)")
    _save_csv(tern_df.set_index(["feature", "value"]), "06_ternary_churn_rates.csv")
    _save_csv(churn_internet.round(4), "06_churn_internet_service.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Gráfico 1: InternetService
    iorder = churn_internet.index.tolist()
    rates  = [churn_internet.loc[c, "churn_rate"] for c in iorder]
    colors = ["tomato" if r > 0.3 else "steelblue" for r in rates]
    bars = axes[0].bar(iorder, rates, color=colors, alpha=0.85)
    for bar, r in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{r*100:.1f}%", ha="center", fontsize=10)
    axes[0].set_title("Churn rate por InternetService\n(justifica one-hot)", fontsize=10)
    axes[0].set_ylabel("Taxa de Churn")
    axes[0].set_ylim(0, 0.55)

    # Gráfico 2: encoding ternário (OnlineSecurity como exemplo)
    sub = tern_df[tern_df["feature"] == "OnlineSecurity"].copy()
    bar_colors = {"No internet service": "lightgray", "No": "orange", "Yes": "steelblue"}
    axes[1].bar(sub["value"], sub["churn_rate"],
                color=[bar_colors[v] for v in sub["value"]], alpha=0.85)
    for _, row in sub.iterrows():
        axes[1].text(row["value"], row["churn_rate"] + 0.005,
                     f"{row['churn_rate']*100:.1f}%", ha="center", fontsize=10)
    axes[1].set_title("Churn rate — OnlineSecurity\n(exemplo do encoding ternário 0/1/2)", fontsize=10)
    axes[1].set_ylabel("Taxa de Churn")
    axes[1].set_ylim(0, 0.55)

    fig.suptitle("Seção 6 — InternetService e encoding ternário", fontsize=12)
    _save_fig(fig, "fig_06_internet_ternario.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 7 — Churn rate por PaymentMethod
# ═══════════════════════════════════════════════════════════════════════════════

def secao_7_payment_method(df: pd.DataFrame) -> None:
    print("\n── Seção 7 · Churn rate por PaymentMethod ───────────────────────")

    churn_pay = (
        df.groupby("PaymentMethod")[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
        .sort_values("churn_rate", ascending=False)
    )
    print(churn_pay.round(4).to_string())
    print("  -> PaymentMethod sem ordenação natural -> one-hot encoding")
    _save_csv(churn_pay.round(4), "07_churn_por_payment.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [l.replace(" (", "\n(") for l in churn_pay.index]
    rates  = churn_pay["churn_rate"].values
    colors = ["tomato" if r > 0.3 else "steelblue" for r in rates]
    bars = ax.bar(labels, rates, color=colors, alpha=0.85)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{r*100:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Taxa de Churn")
    ax.set_ylim(0, 0.55)
    ax.set_title("Seção 7 — Churn rate por PaymentMethod\n"
                 "(Electronic check destaca-se; sem ordem natural -> one-hot)", fontsize=11)
    _save_fig(fig, "fig_07_churn_por_payment.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 8 — Correlação das numéricas com Churn + ratio features
# ═══════════════════════════════════════════════════════════════════════════════

def secao_8_correlacoes_e_ratios(df: pd.DataFrame) -> None:
    print("\n── Seção 8 · Correlações numéricas com Churn e ratio features ───")

    df_num = df.copy()
    df_num["monthly_to_total_ratio"] = df_num["MonthlyCharges"] / df_num["TotalCharges"].replace(0, np.nan)
    df_num["total_per_month"]        = df_num["TotalCharges"] / df_num["tenure"].replace(0, np.nan)
    df_num["log_TotalCharges"]       = np.log1p(df_num["TotalCharges"])
    df_num["log_tenure"]             = np.log1p(df_num["tenure"])

    cols_corr = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "log_TotalCharges", "log_tenure",
        "monthly_to_total_ratio", "total_per_month",
        TARGET,
    ]
    corr_with_target = (
        df_num[cols_corr].corr()[[TARGET]]
        .drop(TARGET)
        .sort_values(TARGET, key=abs, ascending=False)
        .round(4)
    )
    print("  Correlação de Pearson com Churn:")
    print(corr_with_target.to_string())
    print("  -> ratio features e log transforms entram em features_to_keep")
    _save_csv(corr_with_target, "08_correlacao_com_churn.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gráfico 1: correlações em barras
    corr_vals = corr_with_target[TARGET]
    colors = ["tomato" if v < 0 else "steelblue" for v in corr_vals]
    axes[0].barh(corr_vals.index, corr_vals.values, color=colors, alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Correlação de Pearson com Churn")
    axes[0].set_title("Correlação das features numéricas\n(incluindo derivadas) com Churn", fontsize=10)

    # Gráfico 2: scatter monthly_to_total_ratio vs Churn
    sample = df_num.dropna(subset=["monthly_to_total_ratio"]).sample(
        n=min(2000, len(df_num)), random_state=42
    )
    churn_no  = sample[sample[TARGET] == 0]["monthly_to_total_ratio"]
    churn_yes = sample[sample[TARGET] == 1]["monthly_to_total_ratio"]
    axes[1].hist(churn_no,  bins=40, alpha=0.6, color="steelblue", density=True, label="Não cancela")
    axes[1].hist(churn_yes, bins=40, alpha=0.6, color="tomato",    density=True, label="Cancela")
    axes[1].set_xlabel("monthly_to_total_ratio")
    axes[1].set_ylabel("Densidade")
    axes[1].set_title("monthly_to_total_ratio por classe\n"
                       "(clientes novos têm ratio alto = mais propensos a churn)", fontsize=10)
    axes[1].legend()

    fig.suptitle("Seção 8 — Correlações e justificativa das ratio features", fontsize=12)
    _save_fig(fig, "fig_08_correlacoes_e_ratios.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 9 — Heatmap de correlação entre features numéricas derivadas
# ═══════════════════════════════════════════════════════════════════════════════

def secao_9_heatmap_correlacao(df: pd.DataFrame) -> None:
    print("\n── Seção 9 · Heatmap de correlação entre features ───────────────")

    df_num = df.copy()
    df_num["monthly_to_total_ratio"] = df_num["MonthlyCharges"] / df_num["TotalCharges"].replace(0, np.nan)
    df_num["total_per_month"]        = df_num["TotalCharges"] / df_num["tenure"].replace(0, np.nan)
    df_num["log_TotalCharges"]       = np.log1p(df_num["TotalCharges"])
    df_num["log_tenure"]             = np.log1p(df_num["tenure"])

    cols = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "log_tenure", "log_TotalCharges",
        "monthly_to_total_ratio", "total_per_month", TARGET,
    ]
    corr = df_num[cols].corr().round(2)
    _save_csv(corr, "09_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Seção 9 — Matriz de Correlação (features numéricas + derivadas + target)",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    _save_fig(fig, "fig_09_heatmap_correlacao.png")
    print("  -> multicolinearidade tenure↔TotalCharges esperada; ratio features adicionam info nova")


# ═══════════════════════════════════════════════════════════════════════════════
# Seção 10 — Boxplots numéricos por Churn
# ═══════════════════════════════════════════════════════════════════════════════

def secao_10_boxplots_por_churn(df: pd.DataFrame) -> None:
    print("\n── Seção 10 · Boxplots numéricos por classe Churn ───────────────")

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, col in zip(axes, num_cols):
        data = [
            df.loc[df[TARGET] == 0, col].dropna().values,
            df.loc[df[TARGET] == 1, col].dropna().values,
        ]
        bp = ax.boxplot(data, tick_labels=["Não cancela", "Cancela"],
                        patch_artist=True, notch=False,
                        boxprops=dict(alpha=0.7),
                        medianprops=dict(color="crimson", linewidth=2),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("tomato")
        ax.set_title(col, fontsize=11)
        ax.set_ylabel(col)

    fig.suptitle("Seção 10 — Distribuição por Churn: tenure menor e MonthlyCharges maior\n"
                 "justificam ratio features e log transforms", fontsize=11)
    _save_fig(fig, "fig_10_boxplots_por_churn.png")
    print("  -> clientes que cancelam têm tenure menor e MonthlyCharges maior")
    print("  -> confirma utilidade de monthly_to_total_ratio e total_per_month")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    print("=" * 65)
    print("  EDA — Telco Customer Churn")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {DATA_PATH}\n"
            "Execute a ingestão primeiro (src/ingestion.py)."
        )

    df_raw = pd.read_parquet(DATA_PATH)
    df     = secao_1_visao_geral(df_raw)

    secao_2_ausentes_e_flag(df)
    secao_3_target(df)
    secao_4_distribuicoes_numericas(df)
    secao_5_churn_por_contrato(df)
    secao_6_internet_e_ternarios(df)
    secao_7_payment_method(df)
    secao_8_correlacoes_e_ratios(df)
    secao_9_heatmap_correlacao(df)
    secao_10_boxplots_por_churn(df)

    print(f"\n{'=' * 65}")
    print(f"  EDA concluído.")
    print(f"  Stats  -> {STATS_DIR}")
    print(f"  Figs   -> {FIGURES_DIR}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run()
