"""
transformers/type_cast.py — Conversor de tipos de colunas.

Converte colunas para o dtype especificado no YAML (ex: TotalCharges object → float64).
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class TypeCastTransformer(BaseFeatureTransformer):
    """
    Converte colunas para os dtypes configurados em preprocessing.yaml.

    Necessário porque TotalCharges é armazenada como string no CSV original
    e preservada como object no Parquet. pd.to_numeric(..., errors='coerce')
    converte para float64 e substitui valores não numéricos por NaN (os 11
    registros com tenure=0 identificados no EDA).

    Config (preprocessing.yaml → type_cast):
        - column: "TotalCharges"
          dtype: "float64"

    Exemplo:
        transformer = TypeCastTransformer(casts=config['type_cast'], logger=logger)
        df = transformer.fit_transform(df)
    """

    _NUMERIC_DTYPES = {"float64", "float32", "int64", "int32"}

    def __init__(self, casts: list[dict], logger: Any = None) -> None:
        self.casts = casts
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Converte cada coluna configurada para o dtype alvo."""
        X = X.copy()

        for spec in self.casts:
            col = spec["column"]
            dtype = spec["dtype"]

            if col not in X.columns:
                self._warn(
                    "TypeCastTransformer: coluna '%s' não encontrada — ignorada.", col
                )
                continue

            n_nan_antes = int(X[col].isna().sum())

            if dtype in self._NUMERIC_DTYPES:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                try:
                    X[col] = X[col].astype(dtype)
                except (ValueError, TypeError) as exc:
                    self._warn(
                        "TypeCastTransformer: falha ao converter '%s' → %s: %s",
                        col, dtype, exc,
                    )
                    continue

            n_nan_depois = int(X[col].isna().sum())
            self._log(
                "TypeCastTransformer: '%s' → %s | NaN: %d → %d",
                col, dtype, n_nan_antes, n_nan_depois,
            )

        return X
