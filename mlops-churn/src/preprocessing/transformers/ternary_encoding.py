"""
transformers/ternary_encoding.py — Encoder ordinal para colunas com "No service".

Mapeia colunas com 3 valores (No service / No / Yes) para escala ordinal 0/1/2.
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class TernaryEncodingTransformer(BaseFeatureTransformer):
    """
    Aplica encoding ordinal 0/1/2 em colunas com três estados de serviço.

    Contexto do dataset Telco Churn:
      MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
      TechSupport, StreamingTV e StreamingMovies possuem três valores:
        "No phone service" / "No internet service" → 0 (sem acesso ao serviço base)
        "No"                                        → 1 (tem acesso, não contratou o add-on)
        "Yes"                                       → 2 (tem acesso e contratou o add-on)

    Por que ordinal e não one-hot?
    - EDA (Seção 6) confirmou hierarquia de churn:
        No service (7.4%) < Yes (14.6%) < No (41.8%)
    - A ordenação é semanticamente válida: sem serviço < sem feature < com feature.
    - One-hot dobraria o número de colunas sem ganho de informação.

    Config (preprocessing.yaml → ternary_encoding):
        columns:
          - "MultipleLines"
          - "OnlineSecurity"
        ordinal_map:
          "No phone service": 0
          "No internet service": 0
          "No": 1
          "Yes": 2

    Exemplo:
        transformer = TernaryEncodingTransformer(config=config['ternary_encoding'], logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, config: dict, logger: Any = None) -> None:
        self.config = config
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica o mapeamento ordinal em cada coluna configurada."""
        X = X.copy()
        columns = self.config.get("columns", [])
        ordinal_map: dict = self.config.get("ordinal_map", {})

        for col in columns:
            if col not in X.columns:
                self._warn(
                    "TernaryEncodingTransformer: '%s' não encontrada — ignorada.", col
                )
                continue

            X[col] = X[col].map(ordinal_map)
            n_nan = int(X[col].isna().sum())
            if n_nan > 0:
                self._warn(
                    "TernaryEncodingTransformer: '%s' — %d valores sem mapeamento → NaN",
                    col, n_nan,
                )
            n_distinct = len(set(ordinal_map.values()))
            self._log(
                "TernaryEncodingTransformer: '%s' — %d chaves mapeadas para %d valores distintos %s",
                col, len(ordinal_map), n_distinct, sorted(set(ordinal_map.values())),
            )

        return X
