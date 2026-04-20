"""
transformers/derived_features.py — Features derivadas para Telco Churn.

Cria três grupos de features com base em domínio de negócio:

  n_services      → número de add-ons contratados (segurança, backup, streaming, etc.)
                    Clientes com mais serviços têm menor probabilidade de churn.

  tenure_bin      → faixa de maturidade do cliente (ordinal):
                    0=novo (0), 1=iniciante (1-12), 2=médio (13-24),
                    3=estabelecido (25-48), 4=fiel (49+).
                    Novas assinaturas têm churn muito maior — a faixa captura
                    esse efeito de forma mais expressiva que tenure contínuo.

  has_fiber_monthly → flag de alto risco: fibra óptica + contrato mês-a-mês.
                      Essa combinação concentra a maioria dos churners.

  avg_monthly_per_service → MonthlyCharges / n_services.
                            Clientes que pagam mais por add-on têm mais a perder ao sair;
                            valores altos indicam poucos serviços com cobrança elevada.
                            n_services=0 → fill 0 (sem add-ons contratados).

Stateless: fit() é no-op — sem risco de data leakage.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class DerivedFeaturesTransformer(BaseFeatureTransformer):
    """
    Cria features de domínio derivadas das colunas já encodadas.

    Deve ser aplicado APÓS os encodings (ternário, categórico, one-hot),
    pois depende das colunas já transformadas.

    Config (preprocessing.yaml → derived_features):

        service_columns: lista de colunas ternárias de serviço (valor 2 = "Yes")
        tenure_column: coluna de tenure (default "tenure")
        tenure_bins: lista de bordas [0, 12, 24, 48, 72] (default acima)
        fiber_column: coluna one-hot de fibra óptica (default "internet_Fiber optic")
        contract_column: coluna ordinal de contrato (default "contract_encoded")
    """

    def __init__(
        self,
        service_columns: list[str],
        tenure_column: str = "tenure",
        tenure_bins: list[int] | None = None,
        fiber_column: str = "internet_Fiber optic",
        contract_column: str = "contract_encoded",
        monthly_charges_column: str = "MonthlyCharges",
        logger: Any = None,
    ) -> None:
        self.service_columns = service_columns
        self.tenure_column = tenure_column
        self.tenure_bins = tenure_bins or [0, 12, 24, 48, 72]
        self.fiber_column = fiber_column
        self.contract_column = contract_column
        self.monthly_charges_column = monthly_charges_column
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()

        # ── n_services: conta add-ons com valor == 2 ("Yes" no encoding ternário)
        cols_presentes = [c for c in self.service_columns if c in X.columns]
        if cols_presentes:
            X["n_services"] = (X[cols_presentes] == 2).sum(axis=1).astype(np.int8)
            self._log(
                "DerivedFeaturesTransformer: n_services criado (%d colunas): min=%d max=%d",
                len(cols_presentes), int(X["n_services"].min()), int(X["n_services"].max()),
            )

            # ── avg_monthly_per_service: MonthlyCharges / n_services
            if self.monthly_charges_column in X.columns:
                X["avg_monthly_per_service"] = (
                    X[self.monthly_charges_column] / X["n_services"].replace(0, np.nan)
                ).fillna(0.0)
                self._log(
                    "DerivedFeaturesTransformer: avg_monthly_per_service criado "
                    "(média=%.2f, zeros=%d).",
                    float(X["avg_monthly_per_service"].mean()),
                    int((X["avg_monthly_per_service"] == 0).sum()),
                )
            else:
                self._warn(
                    "DerivedFeaturesTransformer: coluna '%s' ausente — "
                    "avg_monthly_per_service ignorado.",
                    self.monthly_charges_column,
                )
        else:
            self._warn("DerivedFeaturesTransformer: nenhuma service_column encontrada.")

        # ── tenure_bin: faixa ordinal de maturidade do cliente
        if self.tenure_column in X.columns:
            bins = self.tenure_bins
            labels = list(range(len(bins) - 1))
            X["tenure_bin"] = pd.cut(
                X[self.tenure_column],
                bins=[-1] + bins[1:],     # -1 inclui tenure=0 no primeiro bin
                labels=labels,
                right=True,
            ).astype(np.int8)
            self._log(
                "DerivedFeaturesTransformer: tenure_bin criado (bins=%s).", bins,
            )
        else:
            self._warn(
                "DerivedFeaturesTransformer: coluna '%s' ausente — tenure_bin ignorado.",
                self.tenure_column,
            )

        # ── has_fiber_monthly: fibra óptica + contrato mês-a-mês = alto risco de churn
        if self.fiber_column in X.columns and self.contract_column in X.columns:
            X["has_fiber_monthly"] = (
                (X[self.fiber_column] == 1) & (X[self.contract_column] == 0)
            ).astype(np.int8)
            pct = float(X["has_fiber_monthly"].mean()) * 100
            self._log(
                "DerivedFeaturesTransformer: has_fiber_monthly criado (%.1f%% dos clientes).", pct,
            )
        else:
            self._warn(
                "DerivedFeaturesTransformer: colunas '%s' ou '%s' ausentes — has_fiber_monthly ignorado.",
                self.fiber_column, self.contract_column,
            )

        return X
