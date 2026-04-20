"""
transformers/categorical_encoder.py — Encoder categórico genérico.

Aplica diferentes estratégias de encoding para variáveis categóricas do dataset
Telco Churn, guiadas inteiramente pelo preprocessing.yaml.

Estratégias suportadas por spec:
  binary  — coluna de 2 categorias → nova coluna 0/1 (ex: gender → is_female)
  ordinal — mapa configurado → inteiro (ex: Contract → 0/1/2)
  one_hot — pd.get_dummies com prefixo configurado (ex: InternetService, PaymentMethod)

Em todos os casos, a coluna original é removida após o encoding.
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class CategoricalEncoder(BaseFeatureTransformer):
    """
    Encoding genérico de variáveis categóricas guiado por lista de specs do YAML.

    Config (preprocessing.yaml → categorical_encoding):
        - column: "gender"
          strategy: "binary"
          positive_value: "Female"
          new_column: "is_female"

        - column: "Contract"
          strategy: "ordinal"
          new_column: "contract_encoded"
          ordinal_map: {"Month-to-month": 0, "One year": 1, "Two year": 2}

        - column: "InternetService"
          strategy: "one_hot"
          prefix: "internet"
          drop_first: false

        - column: "PaymentMethod"
          strategy: "one_hot"
          prefix: "payment"
          drop_first: false

    Exemplo:
        encoder = CategoricalEncoder(encodings=config['categorical_encoding'], logger=logger)
        df = encoder.fit_transform(df)
    """

    def __init__(self, encodings: list[dict], logger: Any = None) -> None:
        self.encodings = encodings
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica cada spec de encoding em sequência; remove a coluna original."""
        X = X.copy()

        for spec in self.encodings:
            col = spec.get("column")
            strategy = spec.get("strategy")

            if col not in X.columns:
                self._warn(
                    "CategoricalEncoder: coluna '%s' não encontrada — spec ignorada.", col
                )
                continue

            if strategy == "binary":
                positive_value = spec["positive_value"]
                new_col = spec["new_column"]
                X[new_col] = (X[col] == positive_value).astype(int)
                X = X.drop(columns=[col])
                self._log(
                    "CategoricalEncoder: '%s' → '%s' (binary, positive='%s')",
                    col, new_col, positive_value,
                )

            elif strategy == "ordinal":
                ordinal_map: dict = spec["ordinal_map"]
                new_col = spec["new_column"]
                X[new_col] = X[col].map(ordinal_map)
                n_nan = int(X[new_col].isna().sum())
                if n_nan > 0:
                    self._warn(
                        "CategoricalEncoder: '%s' ordinal — %d valores não mapeados → NaN",
                        col, n_nan,
                    )
                X = X.drop(columns=[col])
                self._log(
                    "CategoricalEncoder: '%s' → '%s' (ordinal, %d categorias)",
                    col, new_col, len(ordinal_map),
                )

            elif strategy == "one_hot":
                prefix: str = spec.get("prefix", col.lower().replace(" ", "_"))
                drop_first: bool = spec.get("drop_first", False)
                dummies = pd.get_dummies(
                    X[col], prefix=prefix, drop_first=drop_first
                ).astype(int)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                self._log(
                    "CategoricalEncoder: '%s' → one_hot %s",
                    col, list(dummies.columns),
                )

            else:
                self._warn(
                    "CategoricalEncoder: estratégia '%s' desconhecida para '%s' — ignorada.",
                    strategy, col,
                )

        return X
