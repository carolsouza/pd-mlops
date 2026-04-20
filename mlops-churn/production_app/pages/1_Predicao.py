"""
pages/1_Predicao.py — Predição de Churn para cliente Telco.

O usuário preenche os dados do cliente. Ao submeter:
  1. O pipeline de pré-processamento (utils/pipeline_utils.py) converte as entradas
     brutas nas 33 features engenheiradas que o modelo espera.
  2. O modelo é carregado do banco SQLite do MLflow (sem servidor REST).
  3. predict_proba() retorna a probabilidade de churn.
  4. O resultado é exibido com nível de risco, gauge e métricas do modelo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

_PAGE_DIR     = Path(__file__).resolve().parent
_APP_DIR      = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import preprocessar_entradas
from utils.model_utils import carregar_modelo, prever_individual, obter_metricas_modelo

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Churn",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Predição de Churn — Telco Customer")
st.markdown(
    "Preencha os dados do cliente. O pipeline de feature engineering é executado "
    "automaticamente e o modelo é carregado diretamente do **banco SQLite do MLflow**."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
_URI_PADRAO = f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"

with st.sidebar:
    st.header("⚙️ Configurações MLflow")
    db_uri = st.text_input("URI do banco SQLite", value=_URI_PADRAO)
    st.divider()
    st.markdown(
        """
        **Como gerar o banco:**
        ```bash
        python notebooks/preprocessamento.py
        python notebooks/modelagem.py
        ```
        **Executar o app:**
        ```bash
        streamlit run production_app/app.py
        ```
        """
    )


@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def _modelo_em_cache(uri: str):
    return carregar_modelo(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Formulário
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Dados do cliente")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Dados pessoais**")
    gender = st.selectbox("Gênero", ["Male", "Female"])
    senior = st.checkbox("Cliente sênior (65+)")
    partner = st.selectbox("Possui parceiro(a)?", ["No", "Yes"])
    dependents = st.selectbox("Possui dependentes?", ["No", "Yes"])

    st.markdown("**Conta**")
    tenure = st.slider("Tempo de contrato (meses)", 0, 72, 12)
    contract = st.selectbox(
        "Tipo de contrato",
        ["Month-to-month", "One year", "Two year"],
    )
    paperless = st.selectbox("Fatura sem papel?", ["No", "Yes"])
    payment = st.selectbox(
        "Método de pagamento",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input(
        "Cobrança mensal (USD)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
    )

with col2:
    st.markdown("**Serviços de telefonia**")
    phone = st.selectbox("Serviço telefônico?", ["Yes", "No"])
    if phone == "Yes":
        multiple = st.selectbox("Múltiplas linhas?", ["No", "Yes"])
    else:
        multiple = "No phone service"
        st.info("Sem serviço telefônico → múltiplas linhas: N/A")

    st.markdown("**Internet**")
    internet = st.selectbox("Serviço de internet", ["DSL", "Fiber optic", "No"])

with col3:
    st.markdown("**Add-ons de internet**")
    if internet != "No":
        security   = st.selectbox("Segurança online",   ["No", "Yes"], key="sec")
        backup     = st.selectbox("Backup online",       ["No", "Yes"], key="bkp")
        protection = st.selectbox("Proteção de device",  ["No", "Yes"], key="prt")
        support    = st.selectbox("Suporte técnico",     ["No", "Yes"], key="sup")
        tv         = st.selectbox("Streaming TV",        ["No", "Yes"], key="tv")
        movies     = st.selectbox("Streaming filmes",    ["No", "Yes"], key="mov")
    else:
        security = backup = protection = support = tv = movies = "No internet service"
        st.info("Sem internet → add-ons não disponíveis.")

# ─────────────────────────────────────────────────────────────────────────────
# Predição
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
btn = st.button("🔮 Calcular Probabilidade de Churn", type="primary", use_container_width=True)

if btn:
    entradas = {
        "gender"           : gender,
        "SeniorCitizen"    : 1 if senior else 0,
        "Partner"          : partner,
        "Dependents"       : dependents,
        "tenure"           : tenure,
        "PhoneService"     : phone,
        "MultipleLines"    : multiple,
        "InternetService"  : internet,
        "OnlineSecurity"   : security,
        "OnlineBackup"     : backup,
        "DeviceProtection" : protection,
        "TechSupport"      : support,
        "StreamingTV"      : tv,
        "StreamingMovies"  : movies,
        "Contract"         : contract,
        "PaperlessBilling" : paperless,
        "PaymentMethod"    : payment,
        "MonthlyCharges"   : monthly,
    }

    # ── Passo 1: pré-processamento ────────────────────────────────────────────
    with st.spinner("Executando pipeline de feature engineering..."):
        try:
            features_df = preprocessar_entradas(entradas)
            prep_ok = True
        except Exception as exc:
            st.error(f"❌ Erro no pré-processamento: {exc}")
            prep_ok = False

    # ── Passo 2: predição ─────────────────────────────────────────────────────
    if prep_ok:
        with st.spinner("Carregando modelo e calculando probabilidade..."):
            try:
                modelo = _modelo_em_cache(db_uri)
                prob_churn, classe = prever_individual(features_df, modelo)
                pred_ok = True
            except Exception as exc:
                st.error(f"❌ Erro na predição: {exc}\n\nVerifique o banco: `{db_uri}`")
                pred_ok = False

    # ── Passo 3: métricas do modelo ───────────────────────────────────────────
    if prep_ok and pred_ok:
        try:
            metricas = obter_metricas_modelo(db_uri)
        except Exception:
            metricas = None

    # ── Passo 4: exibição ─────────────────────────────────────────────────────
    if prep_ok and pred_ok:
        st.divider()
        st.subheader("📊 Resultado da Predição")

        # Nível de risco
        if prob_churn < 0.30:
            risco, cor, emoji = "BAIXO", "green", "🟢"
        elif prob_churn < 0.60:
            risco, cor, emoji = "MÉDIO", "orange", "🟡"
        else:
            risco, cor, emoji = "ALTO", "red", "🔴"

        col_r1, col_r2, col_r3 = st.columns([2, 1, 1])

        with col_r1:
            st.metric(
                label=f"{emoji} Probabilidade de Churn",
                value=f"{prob_churn:.1%}",
                help="Probabilidade estimada pelo modelo de o cliente cancelar o serviço.",
            )
            st.markdown(f"**Nível de risco:** :{cor}[{risco}]")
            st.progress(
                prob_churn,
                text=f"Risco {risco} — {prob_churn:.1%} de probabilidade de churn",
            )
            st.caption(
                "Limiar de classificação: 0.5 → "
                f"{'⚠️ Churn previsto' if classe == 1 else '✅ Sem churn previsto'}"
            )

        with col_r2:
            st.markdown("**Dados fornecidos**")
            st.write(f"- Tempo de contrato: **{tenure} meses**")
            st.write(f"- Contrato: **{contract}**")
            st.write(f"- Internet: **{internet}**")
            st.write(f"- Mensalidade: **USD {monthly:.2f}**")
            tc = tenure * monthly
            st.write(f"- Total estimado: **USD {tc:,.2f}**")

        with col_r3:
            if metricas:
                st.markdown("**Performance do modelo**")
                st.metric("CV AUC", f"{metricas['cv_roc_auc_mean']:.4f}")
                st.metric("Holdout AUC", f"{metricas['holdout_roc_auc']:.4f}")
                st.metric("Holdout Recall", f"{metricas['holdout_recall']:.4f}")
                st.caption(f"Versão: {metricas['versao_modelo']}")
                st.caption(f"Run: `{metricas['run_id'][:8]}...`")

        # ── Inspeção das features ─────────────────────────────────────────────
        with st.expander("🔍 Inspecionar features engenheiradas"):
            st.caption(f"{len(features_df.columns)} features enviadas ao modelo")
            st.dataframe(
                features_df.T.rename(columns={0: "valor"}).style.format("{:.4f}"),
                use_container_width=True,
                height=500,
            )
