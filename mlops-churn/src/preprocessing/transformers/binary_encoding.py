"""
transformers/binary_encoding.py — Encoder de colunas binárias Yes/No.

Converte múltiplas colunas Yes/No (e o target Churn) em 1/0.
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class BinaryEncodingTransformer(BaseFeatureTransformer):
    """
    Converte colunas Yes/No em 1/0 in-place.

    Por que não usar BinaryFlagTransformer?
    - BinaryFlagTransformer cria uma NOVA coluna e mantém a original.
    - Aqui queremos SUBSTITUIR a coluna original (Partner, Churn, etc.).

    Config (preprocessing.yaml → binary_encoding):
        columns:
          - "Partner"
          - "Dependents"
          - "PhoneService"
          - "PaperlessBilling"
          - "Churn"
        positive_value: "Yes"   # "Yes" → 1, qualquer outro → 0

    Exemplo:
        transformer = BinaryEncodingTransformer(config=config['binary_encoding'], logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, config: dict, logger: Any = None) -> None:
        self.config = config
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Substitui cada coluna configurada por 1 (positive_value) ou 0."""
        X = X.copy()
        columns = self.config.get("columns", [])
        positive_value = self.config.get("positive_value", "Yes")

        for col in columns:
            if col not in X.columns:
                self._warn(
                    "BinaryEncodingTransformer: '%s' não encontrada — ignorada.", col
                )
                continue

            X[col] = (X[col] == positive_value).astype(int)
            self._log(
                "BinaryEncodingTransformer: '%s' → 1 se == '%s', 0 caso contrário",
                col, positive_value,
            )

        return X
