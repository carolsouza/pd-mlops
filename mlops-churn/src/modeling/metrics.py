"""
modeling/metrics.py — Funções de cálculo e agregação de métricas de classificação.

Responsabilidade única: computar e agregar métricas de avaliação.
Nenhuma dependência de MLflow, Optuna ou sklearn.Pipeline — funções puras.

Métricas primária: ROC-AUC (robusta ao desbalanceamento 73%/27%).
Métricas adicionais: F1, Precision, Recall, Accuracy (todas com threshold=0.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def calcular_metricas(
    y_verdadeiro: np.ndarray,
    y_previsto: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Calcula as métricas de classificação usadas no pipeline.

    Parâmetros
    ----------
    y_verdadeiro : array com os valores reais do target (0/1)
    y_previsto   : array com as classes preditas (threshold=0.5)
    y_prob       : array com as probabilidades da classe positiva (para ROC-AUC).
                   Se None, usa y_previsto diretamente (apenas para predict sem proba).

    Retorna
    -------
    dict com chaves: roc_auc, f1, precision, recall, accuracy
    """
    y_score = y_prob if y_prob is not None else y_previsto
    roc_auc   = float(roc_auc_score(y_verdadeiro, y_score))
    f1        = float(f1_score(y_verdadeiro, y_previsto, zero_division=0))
    precision = float(precision_score(y_verdadeiro, y_previsto, zero_division=0))
    recall    = float(recall_score(y_verdadeiro, y_previsto, zero_division=0))
    accuracy  = float(accuracy_score(y_verdadeiro, y_previsto))
    return {
        'roc_auc'  : roc_auc,
        'f1'       : f1,
        'precision': precision,
        'recall'   : recall,
        'accuracy' : accuracy,
    }


def agregar_metricas_folds(fold_metrics: list[dict]) -> dict:
    """
    Agrega métricas de todos os folds em média ± desvio padrão.

    Parâmetros
    ----------
    fold_metrics : lista de dicts com chaves roc_auc, f1, precision, recall, accuracy (e fold)

    Retorna
    -------
    dict com cv_{metrica}_mean e cv_{metrica}_std para cada métrica
    """
    df = pd.DataFrame(fold_metrics)
    resultado = {}
    for col in ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']:
        resultado[f'cv_{col}_mean'] = float(df[col].mean())
        resultado[f'cv_{col}_std']  = float(df[col].std())
    return resultado
