"""
modeling/artifacts.py — Gerador de artefatos diagnósticos do melhor modelo.

Responsabilidade única: salvar plots PNG em disco e retornar seus caminhos.
O MLflowTracker é responsável por logá-los no MLflow — separação de concerns.

Plots gerados (configuráveis em modeling.yaml → artifacts.plots):
  confusion_matrix          — VP/VN/FP/FN com threshold=0.5
  roc_curve                 — curva ROC com AUC (threshold-free)
  precision_recall_curve    — mais informativa que ROC em dados desbalanceados
  feature_importance        — top-20 features por importância (tree, linear ou permutation)
  cv_fold_comparison        — ROC-AUC e F1 por fold (robustez)
  calibration_curve         — calibração das probabilidades preditas
  optuna_history            — histórico de otimização por trial (Optuna)
  optuna_param_importances  — importância dos hiperparâmetros no Optuna
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')   # backend não-interativo: salva em arquivo sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline as SklearnPipeline


class ArtifactGenerator:
    """
    Gera e salva plots diagnósticos do melhor modelo.

    Parâmetros
    ----------
    output_dir   : diretório onde os PNG serão salvos
    artifacts_cfg: dict da seção artifacts em modeling.yaml
    logger       : Logger opcional
    """

    def __init__(
        self,
        output_dir: Path,
        artifacts_cfg: dict,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_dir    = Path(output_dir)
        self.artifacts_cfg = artifacts_cfg
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _salvar(self, fig: Any, nome: str) -> Path | None:
        """Salva a figura em disco e fecha. Retorna o Path ou None em caso de erro."""
        caminho = self.output_dir / nome
        try:
            fig.savefig(caminho, dpi=120, bbox_inches='tight')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Falha ao salvar plot %s: %s', nome, exc)
            return None
        finally:
            plt.close(fig)
        if self.logger:
            self.logger.info('Plot salvo: %s', caminho)
        return caminho

    def _extrair_importancia_features(
        self,
        model: Any,
        feature_names: list[str],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> pd.Series:
        """
        Extrai importância de features do modelo treinado.

        Prioridade:
          1. feature_importances_ (árvores, ensembles baseados em árvores)
          2. coef_                (modelos lineares — usa valor absoluto)
          3. permutation_importance (fallback model-agnóstico: SVC, KNN, ensembles)
        """
        if isinstance(model, SklearnPipeline):
            estimador = model.named_steps['estimator']
            reducer   = model.named_steps.get('reducer')
            if reducer is not None and reducer.selected_features is not None:
                nomes_imp = reducer.selected_features
            else:
                nomes_imp = feature_names
        else:
            estimador = model
            nomes_imp = feature_names

        if hasattr(estimador, 'feature_importances_'):
            return pd.Series(estimador.feature_importances_, index=nomes_imp)
        elif hasattr(estimador, 'coef_'):
            coef = np.abs(estimador.coef_)
            if coef.ndim > 1:
                coef = coef.flatten()
            return pd.Series(coef, index=nomes_imp)
        else:
            amostra = min(2000, len(X_val))
            idx = np.random.default_rng(42).choice(len(X_val), amostra, replace=False)
            r = permutation_importance(
                model, X_val.iloc[idx], y_val.iloc[idx],
                n_repeats=5, random_state=42, n_jobs=-1,
                scoring='roc_auc',
            )
            return pd.Series(r.importances_mean, index=feature_names)

    def _obter_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray | None:
        """Retorna probabilidades da classe positiva (coluna 1) ou None."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        return None

    # ── Plots individuais ─────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> Path | None:
        """Matriz de confusão com threshold=0.5."""
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nao Churn', 'Churn'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'Matriz de Confusao — {model_name}')
        plt.tight_layout()
        return self._salvar(fig, 'confusion_matrix.png')

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
    ) -> Path | None:
        """Curva ROC com area sob a curva (AUC)."""
        fig, ax = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Baseline (AUC=0.50)')
        ax.set_title(f'Curva ROC — {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._salvar(fig, 'roc_curve.png')

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
    ) -> Path | None:
        """Curva Precision-Recall — mais informativa que ROC em dados desbalanceados."""
        fig, ax = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
        prevalencia = float(y_true.mean())
        ax.axhline(prevalencia, color='crimson', linewidth=1.2, linestyle='--',
                   label=f'Baseline (prevalencia={prevalencia:.2f})')
        ax.set_title(f'Curva Precision-Recall — {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._salvar(fig, 'precision_recall_curve.png')

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
    ) -> Path | None:
        """Calibracao das probabilidades preditas — ideal = diagonal."""
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(prob_pred, prob_true, 's-', color='steelblue', label=model_name)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Calibracao perfeita')
            ax.set_xlabel('Probabilidade media predita')
            ax.set_ylabel('Fracao de positivos')
            ax.set_title(f'Curva de Calibracao — {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return self._salvar(fig, 'calibration_curve.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('calibration_curve falhou: %s', exc)
            return None

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
    ) -> Path | None:
        """Top-20 features por importância."""
        try:
            importancia = self._extrair_importancia_features(model, feature_names, X_train, y_train)
            top20 = importancia.nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(10, 8))
            top20.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
            ax.set_xlabel('Importancia')
            ax.set_title(f'Top-20 Features — {model_name}')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            if self.logger:
                self.logger.info('Top-5 features: %s', importancia.nlargest(5).index.tolist())
            return self._salvar(fig, 'feature_importance.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Feature importance falhou: %s', exc)
            return None

    def plot_cv_fold_comparison(
        self, fold_metrics: list[dict], model_name: str
    ) -> Path | None:
        """ROC-AUC e F1 por fold — avalia robustez do modelo."""
        fold_auc = [fm['roc_auc'] for fm in fold_metrics]
        fold_f1  = [fm['f1']      for fm in fold_metrics]
        labels   = [f'Fold {fm["fold"]}' for fm in fold_metrics]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(labels, fold_auc, color='steelblue', edgecolor='white')
        axes[0].axhline(float(np.mean(fold_auc)), color='crimson', linewidth=1.5,
                        linestyle='--', label=f'Media: {np.mean(fold_auc):.4f}')
        axes[0].set_ylabel('ROC-AUC')
        axes[0].set_ylim(0, 1)
        axes[0].set_title('ROC-AUC por Fold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(labels, fold_f1, color='teal', edgecolor='white')
        axes[1].axhline(float(np.mean(fold_f1)), color='crimson', linewidth=1.5,
                        linestyle='--', label=f'Media: {np.mean(fold_f1):.4f}')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_ylim(0, 1)
        axes[1].set_title('F1 por Fold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        fig.suptitle(f'Robustez por Fold — {model_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._salvar(fig, 'cv_fold_comparison.png')

    def plot_holdout_evaluation(
        self,
        y_holdout: pd.Series,
        y_pred_holdout: np.ndarray,
        holdout_metrics: dict,
        model_name: str,
        y_prob_holdout: np.ndarray | None = None,
    ) -> Path | None:
        """Matriz de confusao + curva ROC no holdout."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix
        cm = confusion_matrix(y_holdout.values, y_pred_holdout)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nao Churn', 'Churn'])
        disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
        axes[0].set_title(
            f'Holdout — Matriz de Confusao\nF1={holdout_metrics["f1"]:.4f}  '
            f'Recall={holdout_metrics["recall"]:.4f}'
        )

        # ROC curve (se probabilidades disponíveis)
        if y_prob_holdout is not None:
            RocCurveDisplay.from_predictions(
                y_holdout.values, y_prob_holdout, ax=axes[1], name=model_name
            )
            axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
            axes[1].set_title(
                f'Holdout — Curva ROC\nAUC={holdout_metrics["roc_auc"]:.4f}'
            )
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'predict_proba nao disponivel',
                         ha='center', va='center', transform=axes[1].transAxes)

        fig.suptitle(f'Avaliacao Holdout — {model_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._salvar(fig, 'holdout_evaluation.png')

    def plot_optuna_history(self, study: Any, model_name: str) -> Path | None:
        """Histórico de otimização por trial do Optuna."""
        if len(study.trials) <= 1:
            return None
        try:
            import optuna
            fig_ax = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig_ax.figure.set_size_inches(10, 5)
            return self._salvar(fig_ax.figure, f'optuna_history_{model_name}.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Plot optuna_history falhou para %s: %s', model_name, exc)
            return None

    def plot_optuna_param_importances(self, study: Any, model_name: str) -> Path | None:
        """Importância dos hiperparâmetros segundo o Optuna."""
        if len(study.trials) <= 1:
            return None
        try:
            import optuna
            fig_ax = optuna.visualization.matplotlib.plot_param_importances(study)
            fig_ax.figure.set_size_inches(10, 5)
            return self._salvar(fig_ax.figure, f'optuna_params_{model_name}.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Plot optuna_params falhou para %s: %s', model_name, exc)
            return None

    # ── API de alto nível ─────────────────────────────────────────────────────

    def gerar_diagnosticos_modelo(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fold_metrics: list[dict],
    ) -> tuple[dict[str, Path | None], dict]:
        """
        Gera todos os plots diagnósticos do modelo treinado no conjunto de treino.

        Retorna
        -------
        (caminhos, metricas_treino) onde caminhos mapeia nome_do_plot → Path
        """
        from src.modeling.metrics import calcular_metricas

        y_pred_train = model.predict(X_train)
        y_prob_train = self._obter_proba(model, X_train)
        metricas_treino = calcular_metricas(y_train.values, y_pred_train, y_prob_train)

        plots_habilitados = self.artifacts_cfg.get('plots', [])
        caminhos: dict[str, Path | None] = {}

        if 'confusion_matrix' in plots_habilitados:
            caminhos['confusion_matrix'] = self.plot_confusion_matrix(
                y_train.values, y_pred_train, model_name
            )

        if 'roc_curve' in plots_habilitados and y_prob_train is not None:
            caminhos['roc_curve'] = self.plot_roc_curve(
                y_train.values, y_prob_train, model_name
            )

        if 'precision_recall_curve' in plots_habilitados and y_prob_train is not None:
            caminhos['precision_recall_curve'] = self.plot_precision_recall_curve(
                y_train.values, y_prob_train, model_name
            )

        if 'calibration_curve' in plots_habilitados and y_prob_train is not None:
            caminhos['calibration_curve'] = self.plot_calibration_curve(
                y_train.values, y_prob_train, model_name
            )

        if 'feature_importance' in plots_habilitados:
            caminhos['feature_importance'] = self.plot_feature_importance(
                model, list(X_train.columns), X_train, y_train, model_name
            )

        if 'cv_fold_comparison' in plots_habilitados:
            caminhos['cv_fold_comparison'] = self.plot_cv_fold_comparison(fold_metrics, model_name)

        return caminhos, metricas_treino
