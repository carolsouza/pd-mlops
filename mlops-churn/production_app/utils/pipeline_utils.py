"""
production_app/utils/pipeline_utils.py — Pré-processamento para inferência Telco Churn.

Replica a cadeia completa de preprocessing.py usando as mesmas classes e config YAML.
Todos os transformadores são stateless (sem fit em dados de treino), então podem ser
aplicados diretamente sobre a entrada do usuário.

Entrada bruta (campos preenchidos pelo usuário na UI):
    gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod, MonthlyCharges

TotalCharges: calculado automaticamente como tenure × MonthlyCharges.
customerID: preenchido com placeholder (removido pelo FeatureSelector).

Saída: DataFrame de 1 linha com as colunas exatas que o modelo espera.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_HERE         = Path(__file__).resolve().parent
_APP_DIR      = _HERE.parent
_PROJECT_ROOT = _APP_DIR.parent
_CONFIG_DIR   = _PROJECT_ROOT / "config"
_DATA_DIR     = _PROJECT_ROOT / "data"

for _p in [str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.preprocessing.pipeline_builder import PreprocessingPipelineBuilder


def _carregar_cfg() -> dict[str, Any]:
    with open(_CONFIG_DIR / "preprocessing.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_CFG = _carregar_cfg()
_TARGET_COL: str = _CFG.get("feature_selection", {}).get("target", "Churn")
_FEATURES_TO_KEEP: list[str] = [
    c for c in _CFG.get("feature_selection", {}).get("features_to_keep", [])
    if c != _TARGET_COL
]

_PARQUET_FEATURES = _DATA_DIR / "features" / "telco_customer_churn_features.parquet"


def preprocessar_entradas(raw: dict[str, Any]) -> pd.DataFrame:
    """
    Aplica o pipeline completo de feature engineering sobre a entrada bruta do usuário.

    Parâmetros
    ----------
    raw : dict com os campos do formulário Streamlit (sem Churn e sem customerID).

    Retorna
    -------
    pd.DataFrame de 1 linha com as colunas _FEATURES_TO_KEEP sanitizadas para XGBoost.
    """
    # TotalCharges: regra de negócio (tenure=0 → 0; outros → tenure × MonthlyCharges)
    tenure         = int(raw.get("tenure", 0))
    monthly        = float(raw.get("MonthlyCharges", 0.0))
    total_charges  = str(tenure * monthly) if tenure > 0 else "0"

    linha = {
        "customerID"       : "inference-0",
        "gender"           : raw.get("gender", "Male"),
        "SeniorCitizen"    : int(raw.get("SeniorCitizen", 0)),
        "Partner"          : raw.get("Partner", "No"),
        "Dependents"       : raw.get("Dependents", "No"),
        "tenure"           : tenure,
        "PhoneService"     : raw.get("PhoneService", "No"),
        "MultipleLines"    : raw.get("MultipleLines", "No phone service"),
        "InternetService"  : raw.get("InternetService", "No"),
        "OnlineSecurity"   : raw.get("OnlineSecurity", "No internet service"),
        "OnlineBackup"     : raw.get("OnlineBackup", "No internet service"),
        "DeviceProtection" : raw.get("DeviceProtection", "No internet service"),
        "TechSupport"      : raw.get("TechSupport", "No internet service"),
        "StreamingTV"      : raw.get("StreamingTV", "No internet service"),
        "StreamingMovies"  : raw.get("StreamingMovies", "No internet service"),
        "Contract"         : raw.get("Contract", "Month-to-month"),
        "PaperlessBilling" : raw.get("PaperlessBilling", "No"),
        "PaymentMethod"    : raw.get("PaymentMethod", "Electronic check"),
        "MonthlyCharges"   : monthly,
        "TotalCharges"     : total_charges,
    }

    df = pd.DataFrame([linha])
    pipeline = PreprocessingPipelineBuilder(config=_CFG).build()
    df_out = pipeline.fit_transform(df)

    # Garante todas as colunas esperadas (one-hot pode omitir categorias ausentes)
    df_out = df_out.reindex(columns=_FEATURES_TO_KEEP, fill_value=0)

    # Sanitiza nomes para XGBoost
    rename = {
        c: c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in df_out.columns
        if any(ch in c for ch in ("<", "[", "]"))
    }
    if rename:
        df_out = df_out.rename(columns=rename)

    return df_out


def obter_parquet_features() -> pd.DataFrame:
    """Retorna o parquet de features completo (usado pela página de monitoramento)."""
    if not _PARQUET_FEATURES.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado: {_PARQUET_FEATURES}\n"
            "Execute notebooks/preprocessamento.py primeiro."
        )
    return pd.read_parquet(_PARQUET_FEATURES)


def obter_colunas_features() -> list[str]:
    """Retorna nomes das features com sanitização XGBoost."""
    return [
        c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in _FEATURES_TO_KEEP
    ]
