"""
modeling/evaluator.py — Avaliador do modelo no conjunto holdout.

Responsabilidade única: executar a avaliação final em dados nunca vistos
durante o treinamento ou seleção de hiperparâmetros.

O holdout é o "cofre selado" — nunca entrou em nenhum fold de CV,
não influenciou a seleção de hiperparâmetros e não foi visto na escolha
do melhor modelo. É a estimativa mais honesta da performance em produção.
"""
from __future__ import annotations

import logging
from typing import Any

from src.modeling.base import BaseEvaluator
from src.modeling.metrics import calcular_metricas


class HoldoutEvaluator(BaseEvaluator):
    """
    Avalia o modelo final no conjunto holdout.

    Análise de robustez:
      • Holdout AUC ≈ CV AUC     → modelo generaliza bem
      • Holdout AUC << CV AUC    → possível overfitting ou data leakage
      • Diferença < 5pp é considerada aceitável em classificação tabular

    Parâmetros
    ----------
    logger : Logger opcional para diagnósticos de robustez
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger

    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        """
        Calcula métricas do modelo no conjunto holdout.

        Usa predict_proba()[:, 1] para ROC-AUC quando disponível.

        Parâmetros
        ----------
        model : modelo treinado (sklearn Pipeline ou estimador)
        X     : features do holdout
        y     : target do holdout

        Retorna
        -------
        dict com roc_auc, f1, precision, recall, accuracy
        """
        y_pred = model.predict(X)
        y_prob = (
            model.predict_proba(X)[:, 1]
            if hasattr(model, 'predict_proba')
            else None
        )
        return calcular_metricas(y.values, y_pred, y_prob)

    def diagnosticar_robustez(self, cv_auc: float, holdout_auc: float) -> str:
        """
        Compara o CV AUC com o Holdout AUC e emite diagnóstico de robustez.

        Parâmetros
        ----------
        cv_auc      : ROC-AUC médio de cross-validation
        holdout_auc : ROC-AUC no conjunto holdout

        Retorna
        -------
        str com o diagnóstico ('BOA', 'MODERADA' ou 'RUIM')
        """
        delta_pp = cv_auc - holdout_auc   # positivo → AUC caiu no holdout

        if self.logger:
            self.logger.info('── Análise de Robustez ──')
            self.logger.info('  CV AUC (média)   : %.4f', cv_auc)
            self.logger.info('  Holdout AUC      : %.4f', holdout_auc)
            self.logger.info('  Delta             : %.4f pp', delta_pp)

        if delta_pp < 0.05:
            diagnostico = 'BOA'
            if self.logger:
                self.logger.info('  Diagnostico      : Generalizacao BOA (delta < 5pp)')
        elif delta_pp < 0.10:
            diagnostico = 'MODERADA'
            if self.logger:
                self.logger.info('  Diagnostico      : Generalizacao MODERADA (5pp <= delta < 10pp)')
        else:
            diagnostico = 'RUIM'
            if self.logger:
                self.logger.warning('  Diagnostico      : Generalizacao RUIM (delta >= 10pp) -- risco de overfitting!')

        return diagnostico