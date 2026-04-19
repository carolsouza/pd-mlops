"""
transformers/stateful.py — Transformadores Stateful (aprendem parâmetros do treino).

⚠  AVISO MLOps — Data Leakage
   Estes transformadores aprendem estatísticas dos dados de treino no fit().
   NUNCA aplique fit() em todo o dataset antes do split treino/holdout.

   Fluxo correto:
       imputer.fit(X_treino).transform(X_treino)   # aprende no treino
       imputer.transform(X_holdout)                  # aplica no holdout

   Integração no pipeline de modelagem (modelagem.py):
       pipe = Pipeline([
           ('imputer', GroupMedianImputer(...)),    # ← aprende no treino
           ('scaler',  StandardScalerTransformer(...)),
           ('modelo',  Ridge()),
       ])
       pipe.fit(X_treino, y_treino)

   NÃO use estes transformadores no preprocessamento.py — esse script roda
   antes do split e aplicaria fit() em dados de teste, causando data leakage.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class GroupMedianImputer(BaseFeatureTransformer):
    """
    Imputa valores ausentes usando a mediana do grupo ou mediana global.

    Quando group_col=None, usa mediana global de target_col (modo simples).
    Quando group_col é fornecido, usa mediana estratificada por grupo.

    Atributos aprendidos no fit (APENAS no conjunto de treino):
        medians_       (dict | None): {valor_do_grupo → mediana} ou None
        global_median_ (float): mediana global (fallback ou modo principal)

    Raises:
        KeyError:     Se group_col ou target_col não existirem no DataFrame.
        RuntimeError: Se transform() for chamado antes de fit().
    """

    def __init__(
        self,
        target_col: str,
        group_col: str | None = None,
        logger: Any = None,
    ) -> None:
        self.target_col = target_col
        self.group_col = group_col
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "GroupMedianImputer":
        """Aprende mediana(s) de target_col a partir do conjunto de treino."""
        if self.target_col not in X.columns:
            raise KeyError(
                f"GroupMedianImputer.fit: coluna '{self.target_col}' ausente no DataFrame."
            )

        self.global_median_ = float(X[self.target_col].median())

        if self.group_col is not None:
            if self.group_col not in X.columns:
                raise KeyError(
                    f"GroupMedianImputer.fit: coluna de grupo '{self.group_col}' ausente."
                )
            self.medians_ = (
                X.groupby(self.group_col)[self.target_col]
                .median()
                .to_dict()
            )
            self._log(
                "GroupMedianImputer.fit: medianas por '%s' para '%s': %s",
                self.group_col, self.target_col,
                {k: round(v, 1) for k, v in self.medians_.items()},
            )
        else:
            self.medians_ = None
            self._log(
                "GroupMedianImputer.fit: mediana global de '%s' = %.4f",
                self.target_col, self.global_median_,
            )

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Preenche NaN em target_col com a mediana aprendida."""
        if not hasattr(self, "global_median_"):
            raise RuntimeError(
                "GroupMedianImputer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        n_antes = int(X[self.target_col].isna().sum())

        if self.medians_ is not None:
            def _preencher(row: pd.Series) -> float:
                if pd.isna(row[self.target_col]):
                    valor = self.medians_.get(row[self.group_col], self.global_median_)
                    return valor if not pd.isna(valor) else self.global_median_
                return row[self.target_col]

            X[self.target_col] = X.apply(_preencher, axis=1)
        else:
            X[self.target_col] = X[self.target_col].fillna(self.global_median_)

        n_depois = int(X[self.target_col].isna().sum())
        self._log(
            "GroupMedianImputer.transform: '%s' — NaN antes=%d, depois=%d",
            self.target_col, n_antes, n_depois,
        )
        return X


class StandardScalerTransformer(BaseFeatureTransformer):
    """
    Aplica Z-score normalization: z = (x − μ) / σ.

    Por que StandardScaler?
    - Regressão linear e SVM são sensíveis à escala das features.
    - Gradient boosting e Random Forest NÃO precisam de escalonamento.

    Parâmetros aprendidos no fit (APENAS no conjunto de treino):
        mean_  (dict): {coluna: média}
        std_   (dict): {coluna: desvio padrão}

    Colunas com std=0 são ignoradas (constantes — sem informação).

    Raises:
        RuntimeError: Se transform() for chamado antes de fit().
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "StandardScalerTransformer":
        """Aprende média e desvio padrão das colunas especificadas."""
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        ausentes: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                ausentes.append(col)
                continue

            mu = float(X[col].mean())
            sigma = float(X[col].std())

            if sigma == 0:
                self._warn(
                    "StandardScalerTransformer.fit: '%s' tem std=0 (constante) — ignorada.", col
                )
                continue

            self.mean_[col] = mu
            self.std_[col] = sigma

        if ausentes:
            self._warn(
                "StandardScalerTransformer.fit: colunas ausentes ignoradas: %s", ausentes
            )

        self._log(
            "StandardScalerTransformer.fit: parâmetros aprendidos para %d colunas.",
            len(self.mean_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica Z-score nas colunas ajustadas no fit."""
        if not hasattr(self, "mean_"):
            raise RuntimeError(
                "StandardScalerTransformer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        escalonadas: list[str] = []

        for col, mu in self.mean_.items():
            if col not in X.columns:
                continue
            X[col] = (X[col] - mu) / self.std_[col]
            escalonadas.append(col)

        self._log(
            "StandardScalerTransformer.transform: %d colunas escalonadas (z-score).",
            len(escalonadas),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Retorna DataFrame com média e desvio padrão aprendidos (útil para auditoria)."""
        return pd.DataFrame({"mean": self.mean_, "std": self.std_}).rename_axis("feature")
