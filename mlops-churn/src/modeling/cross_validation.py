"""
modeling/cross_validation.py — Executor de Cross-Validation leak-free.

Responsabilidade única: executar StratifiedKFold CV clonando o modelo a cada fold,
garantindo isolação total de estado entre folds e prevenindo data leakage.

O CVRunner é agnóstico ao tipo de modelo — aceita qualquer sklearn estimador
ou Pipeline, incluindo StackingClassifier e VotingClassifier.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.modeling.metrics import calcular_metricas


class CVRunner:
    """
    Executor de Cross-Validation com isolação de estado por fold.

    O clone() do modelo em cada fold garante que:
      - Transformadores stateful (GroupMedianImputer, StandardScalerTransformer)
        aprendem apenas nos índices de treino daquele fold
      - Nenhum parâmetro aprendido é compartilhado entre folds

    StratifiedKFold preserva a proporção de classes (Churn 0/1) em cada fold —
    obrigatório com dados desbalanceados (73%/27%).

    Parâmetros
    ----------
    cv : StratifiedKFold
        Objeto de cross-validation configurado (n_splits, shuffle, random_state)
    logger : logging.Logger, opcional
        Logger para mensagens diagnósticas
    """

    def __init__(self, cv: StratifiedKFold, logger: logging.Logger | None = None) -> None:
        self.cv = cv
        self.logger = logger

    def executar(
        self,
        modelo: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> list[dict]:
        """
        Executa o CV e retorna métricas por fold.

        Clona o modelo em cada fold para evitar contaminação de estado.
        Usa predict_proba()[:, 1] para ROC-AUC (quando disponível).

        Parâmetros
        ----------
        modelo : estimador sklearn ou Pipeline
        X      : features de treino
        y      : target de treino (binário 0/1)

        Retorna
        -------
        Lista de dicts: [{'fold': int, 'roc_auc': float, 'f1': float, ...}, ...]
        """
        metricas_folds = []

        for i_fold, (idx_treino, idx_val) in enumerate(self.cv.split(X, y)):
            m = clone(modelo)
            m.fit(X.iloc[idx_treino], y.iloc[idx_treino])

            y_val     = y.iloc[idx_val].values
            y_pred    = m.predict(X.iloc[idx_val])
            y_prob    = (
                m.predict_proba(X.iloc[idx_val])[:, 1]
                if hasattr(m, 'predict_proba')
                else None
            )

            metricas = calcular_metricas(y_val, y_pred, y_prob)
            metricas['fold'] = i_fold + 1
            metricas_folds.append(metricas)

        return metricas_folds

    @staticmethod
    def de_config(cv_cfg: dict, seed: int) -> "CVRunner":
        """
        Constrói um CVRunner a partir da configuração YAML (seção cv).

        Parâmetros
        ----------
        cv_cfg : dict com chaves n_splits, shuffle
        seed   : semente aleatória global

        Retorna
        -------
        CVRunner configurado
        """
        cv = StratifiedKFold(
            n_splits=cv_cfg.get('n_splits', 5),
            shuffle=cv_cfg.get('shuffle', True),
            random_state=seed,
        )
        return CVRunner(cv=cv)
