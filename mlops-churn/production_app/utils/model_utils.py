"""
production_app/utils/model_utils.py — Cliente MLflow local para Telco Churn.

Carrega o modelo registrado diretamente do banco SQLite via mlflow.sklearn,
expondo predict() e predict_proba() sem necessidade de servidor REST.

Responsabilidades:
1. carregar_modelo(db_uri)      — carrega o Pipeline sklearn do MLflow Registry
2. prever_individual(df, model) — retorna (prob_churn, classe)
3. prever_lote(df, model)       — retorna arrays de probabilidades e classes
4. obter_metricas_modelo(db_uri)— recupera AUC, F1, holdout_auc do MLflow
"""
from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

_NOME_MODELO: str = "telco-churn-best"
_N_FOLDS_CV: int  = 5   # modeling.yaml → cv.n_splits


def carregar_modelo(db_uri: str):
    """Carrega o Pipeline sklearn registrado no MLflow (expõe predict_proba)."""
    mlflow.set_tracking_uri(db_uri)
    return mlflow.sklearn.load_model(f"models:/{_NOME_MODELO}/latest")


def prever_individual(
    features_df: pd.DataFrame,
    modelo: Any,
) -> tuple[float, int]:
    """
    Prediz churn para uma única linha.

    Retorna
    -------
    (prob_churn, classe) — probabilidade [0,1] e classe binária (0 ou 1)
    """
    prob = float(modelo.predict_proba(features_df)[0, 1])
    classe = int(modelo.predict(features_df)[0])
    return prob, classe


def prever_lote(
    features_df: pd.DataFrame,
    modelo: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prediz churn em lote.

    Retorna
    -------
    (probs, classes) — arrays de probabilidades e classes binárias
    """
    probs   = modelo.predict_proba(features_df)[:, 1]
    classes = modelo.predict(features_df)
    return probs, classes


def obter_metricas_modelo(db_uri: str) -> dict[str, Any]:
    """
    Recupera métricas do melhor run registrado no MLflow.

    Retorna
    -------
    dict com cv_roc_auc_mean, cv_roc_auc_std, holdout_roc_auc,
    holdout_f1, holdout_recall, run_id, versao_modelo
    """
    cliente = mlflow.MlflowClient(tracking_uri=db_uri)
    versoes = cliente.search_model_versions(f"name='{_NOME_MODELO}'")

    if not versoes:
        raise ValueError(
            f"Nenhuma versão registrada para '{_NOME_MODELO}'.\n"
            "Execute notebooks/modelagem.py primeiro."
        )

    melhor = max(versoes, key=lambda v: int(v.version))
    run    = cliente.get_run(melhor.run_id)
    m      = run.data.metrics

    return {
        "cv_roc_auc_mean" : float(m.get("cv_roc_auc_mean", 0.0)),
        "cv_roc_auc_std"  : float(m.get("cv_roc_auc_std",  0.0)),
        "holdout_roc_auc" : float(m.get("holdout_roc_auc", 0.0)),
        "holdout_f1"      : float(m.get("holdout_f1",      0.0)),
        "holdout_recall"  : float(m.get("holdout_recall",  0.0)),
        "holdout_precision": float(m.get("holdout_precision", 0.0)),
        "run_id"          : melhor.run_id,
        "versao_modelo"   : melhor.version,
    }
