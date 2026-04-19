"""
pages/2_Monitoramento.py — Dashboard de monitoramento do modelo de churn.

Simula monitoramento em lote:
  1. Amostra N clientes do parquet de features (dados com target real).
  2. Executa predição em lote via modelo MLflow local.
  3. Divide em K lotes sequenciais para simular ingestão temporal.
  4. Calcula por lote: AUC, F1, Precision, Recall, Accuracy e Churn rate.
  5. Visualiza: séries temporais, distribuição de probabilidades, matriz de confusão.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

_PAGE_DIR     = Path(__file__).resolve().parent
_APP_DIR      = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import obter_parquet_features, obter_colunas_features, _TARGET_COL
from utils.model_utils import carregar_modelo, prever_lote

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitoramento do Modelo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dashboard de Monitoramento — Churn Model")
st.markdown(
    "Simula **monitoramento de produção em lote**: amostra N clientes do parquet de features, "
    "executa o modelo e calcula métricas de classificação ao longo de K lotes sequenciais."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
_URI_PADRAO = f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"

with st.sidebar:
    st.header("⚙️ Configurações")
    db_uri = st.text_input("URI do banco SQLite", value=_URI_PADRAO)
    n_amostras = st.slider("Total de amostras", 50, 1000, 300, 50)
    n_lotes    = st.slider("Número de lotes",    5,   50,   20,  5)
    janela     = st.slider("Janela média móvel",  2,   10,   3)
    semente    = st.number_input("Semente aleatória", value=42, step=1)
    threshold  = st.slider(
        "Threshold de classificação", 0.1, 0.9, 0.5, 0.05,
        help="Probabilidade acima da qual o cliente é classificado como Churn.",
    )


@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def _modelo_em_cache(uri: str):
    return carregar_modelo(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metricas_lote(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {
        "auc"       : auc,
        "f1"        : float(f1_score(y_true, y_pred, zero_division=0)),
        "precision" : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"    : float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy"  : float(accuracy_score(y_true, y_pred)),
        "churn_rate_real"   : float(y_true.mean()),
        "churn_rate_previsto": float(y_pred.mean()),
    }


def _serie_temporal(ax, lotes, valores, movel, titulo, cor):
    ax.plot(lotes, valores, "o-", color=cor, alpha=0.45, linewidth=1.5, markersize=4, label="Por lote")
    ax.plot(lotes, movel,   "-",  color=cor, linewidth=2.5, label=f"Média móvel (j={janela})")
    ax.fill_between(lotes, np.array(movel) - np.nanstd(valores),
                    np.array(movel) + np.nanstd(valores), color=cor, alpha=0.08)
    ax.set_title(titulo, fontsize=11, fontweight="bold", color="white")
    ax.set_xlabel("Lote #", fontsize=9, color="white")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("#333")


# ─────────────────────────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────────────────────────
btn = st.button("▶️ Executar Análise de Monitoramento", type="primary", use_container_width=True)

if btn:

    # ── Carrega features ──────────────────────────────────────────────────────
    with st.spinner("Carregando parquet de features..."):
        try:
            df_feat = obter_parquet_features()
        except Exception as exc:
            st.error(f"❌ {exc}")
            st.stop()

    # ── Amostragem ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(int(semente))
    idx = rng.choice(len(df_feat), size=min(n_amostras, len(df_feat)), replace=False)
    df  = df_feat.iloc[idx].reset_index(drop=True)

    y_true = df[_TARGET_COL].values
    cols_feat = obter_colunas_features()
    cols_disp = [c for c in df.columns if c in cols_feat or
                 c.replace("lt_", "<").replace("(", "[").replace(")", "]") in cols_feat]
    X = df[[c for c in df.columns if c != _TARGET_COL]].copy()

    # Sanitiza nomes para XGBoost
    rename = {c: c.replace("<", "lt_").replace("[", "(").replace("]", ")") for c in X.columns
              if any(ch in c for ch in ("<", "[", "]"))}
    if rename:
        X = X.rename(columns=rename)

    # ── Predição em lote ──────────────────────────────────────────────────────
    with st.spinner(f"Predição em lote ({len(X)} clientes)..."):
        try:
            modelo = _modelo_em_cache(db_uri)
            y_prob, _ = prever_lote(X, modelo)
        except Exception as exc:
            st.error(f"❌ Erro na predição: {exc}")
            st.stop()

    y_pred_global = (y_prob >= threshold).astype(int)

    # ── Métricas por lote ─────────────────────────────────────────────────────
    tam   = len(X) // n_lotes
    lotes_mets = []
    for i in range(n_lotes):
        ini = i * tam
        fim = ini + tam + (1 if i < len(X) % n_lotes else 0)
        if fim > len(X):
            break
        m = _metricas_lote(y_true[ini:fim], y_prob[ini:fim], threshold)
        m["lote"] = i + 1
        lotes_mets.append(m)

    df_m  = pd.DataFrame(lotes_mets).set_index("lote")
    df_mv = df_m.rolling(window=janela, min_periods=1).mean()
    geral = _metricas_lote(y_true, y_prob, threshold)
    lotes = df_m.index.tolist()

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 1 — KPIs gerais
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(f"Métricas Gerais — {len(X)} clientes | Threshold = {threshold:.2f}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("ROC-AUC",   f"{geral['auc']:.4f}")
    k2.metric("F1-Score",  f"{geral['f1']:.4f}")
    k3.metric("Recall",    f"{geral['recall']:.4f}")
    k4.metric("Precision", f"{geral['precision']:.4f}")
    k5.metric("Accuracy",  f"{geral['accuracy']:.4f}")

    cr1, cr2 = st.columns(2)
    cr1.metric("Churn rate real",     f"{geral['churn_rate_real']:.1%}")
    cr2.metric("Churn rate previsto", f"{geral['churn_rate_previsto']:.1%}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 2 — Séries temporais
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📈 Métricas por Lote")

    paleta = {"auc": "#3498db", "f1": "#27ae60", "recall": "#e67e22", "precision": "#9b59b6"}
    fig_ts, axs = plt.subplots(2, 2, figsize=(14, 7), tight_layout=True)
    fig_ts.patch.set_facecolor("#0e1117")

    pares = [
        ("auc",       axs[0, 0], "ROC-AUC por Lote"),
        ("f1",        axs[0, 1], "F1-Score por Lote"),
        ("recall",    axs[1, 0], "Recall por Lote"),
        ("precision", axs[1, 1], "Precision por Lote"),
    ]
    for col, ax, titulo in pares:
        _serie_temporal(ax, lotes, df_m[col].tolist(), df_mv[col].tolist(), titulo, paleta[col])

    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 3 — Churn rate real vs previsto por lote
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📉 Churn Rate Real vs Previsto por Lote")

    fig_cr, ax_cr = plt.subplots(figsize=(14, 4), tight_layout=True)
    fig_cr.patch.set_facecolor("#0e1117")
    ax_cr.set_facecolor("#1a1a2e")
    ax_cr.plot(lotes, df_m["churn_rate_real"].tolist(),     "o-", color="#e74c3c",
               label="Real", linewidth=2, markersize=5)
    ax_cr.plot(lotes, df_m["churn_rate_previsto"].tolist(), "s--", color="#3498db",
               label="Previsto", linewidth=2, markersize=5)
    ax_cr.axhline(geral["churn_rate_real"], color="#e74c3c", linestyle=":", alpha=0.5,
                  label=f"Média real: {geral['churn_rate_real']:.1%}")
    ax_cr.set_xlabel("Lote #", color="white")
    ax_cr.set_ylabel("Churn rate", color="white")
    ax_cr.set_title("Churn Rate por Lote", fontsize=12, fontweight="bold", color="white")
    ax_cr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_cr.legend(facecolor="#1a1a2e", labelcolor="white")
    ax_cr.grid(True, linestyle="--", alpha=0.3)
    ax_cr.tick_params(colors="white")
    for s in ax_cr.spines.values():
        s.set_edgecolor("#333")

    st.pyplot(fig_cr, use_container_width=True)
    plt.close(fig_cr)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 4 — Distribuição de probabilidades + Matriz de confusão
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📊 Distribuição de Probabilidades e Matriz de Confusão")

    fig_dist, axs2 = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    fig_dist.patch.set_facecolor("#0e1117")

    # Histograma de probabilidades por classe real
    ax_d = axs2[0]
    ax_d.set_facecolor("#1a1a2e")
    sns.histplot(y_prob[y_true == 0], bins=30, color="#3498db", alpha=0.6,
                 label="Não Churn (real)", ax=ax_d, kde=True)
    sns.histplot(y_prob[y_true == 1], bins=30, color="#e74c3c", alpha=0.6,
                 label="Churn (real)", ax=ax_d, kde=True)
    ax_d.axvline(threshold, color="yellow", linestyle="--", linewidth=1.5,
                 label=f"Threshold = {threshold:.2f}")
    ax_d.set_xlabel("Probabilidade de Churn", color="white")
    ax_d.set_ylabel("Contagem", color="white")
    ax_d.set_title("Distribuição de Probabilidades por Classe", color="white",
                   fontsize=11, fontweight="bold")
    ax_d.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_d.tick_params(colors="white")
    for s in ax_d.spines.values():
        s.set_edgecolor("#333")

    # Matriz de confusão
    ax_cm = axs2[1]
    ax_cm.set_facecolor("#1a1a2e")
    cm = confusion_matrix(y_true, y_pred_global)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não Churn", "Churn"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"Matriz de Confusão (threshold={threshold:.2f})",
        color="white", fontsize=11, fontweight="bold",
    )
    ax_cm.tick_params(colors="white")
    ax_cm.xaxis.label.set_color("white")
    ax_cm.yaxis.label.set_color("white")
    for s in ax_cm.spines.values():
        s.set_edgecolor("#333")

    st.pyplot(fig_dist, use_container_width=True)
    plt.close(fig_dist)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 5 — Tabela de métricas por lote
    # ═══════════════════════════════════════════════════════════════════════════
    with st.expander("📋 Tabela de métricas por lote"):
        df_exib = df_m[["auc", "f1", "recall", "precision", "accuracy",
                         "churn_rate_real", "churn_rate_previsto"]].copy()
        df_exib.columns = ["AUC", "F1", "Recall", "Precision", "Accuracy",
                            "Churn Rate Real", "Churn Rate Previsto"]
        st.dataframe(df_exib.style.format("{:.4f}"), use_container_width=True)
