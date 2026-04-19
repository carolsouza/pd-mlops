"""
preprocessing/pipeline_builder.py — Construtor do Pipeline de Pré-processamento.

Responsabilidade única: ler a configuração do YAML e montar um sklearn.Pipeline
com os transformadores stateless na ordem correta para o dataset Telco Churn.

Princípio de design — Separação entre política e mecanismo:
  • Política  → config/preprocessing.yaml  (O QUÊ transformar e com quais parâmetros)
  • Mecanismo → este arquivo + transformers/ (COMO executar cada transformação)

Ordem do pipeline (dependências entre etapas):
  1. TypeCastTransformer      — TotalCharges object → float64 (deve ser o primeiro)
  2. BinaryFlagTransformer    — is_new_customer (usa tenure original)
  3. ConstantImputer          — imputa TotalCharges NaN com 0 (tenure=0 → sem cobrança)
  4. BinaryEncodingTransformer — Yes/No → 1/0 (Partner, Dependents, Churn, …)
  5. TernaryEncodingTransformer — No service/No/Yes → 0/1/2
  6. CategoricalEncoder       — gender binary, Contract ordinal, InternetService/PaymentMethod one-hot
  7. RatioFeatureTransformer  — monthly_to_total_ratio, total_per_month
  8. LogTransformer           — log_TotalCharges, log_tenure
  9. FeatureSelector          — subconjunto final definido no YAML

⚠  Transformadores stateful (StandardScalerTransformer) NÃO são incluídos aqui.
   Eles devem ser aplicados DENTRO do pipeline de modelagem (modelagem.py),
   APÓS o split treino/holdout, para evitar data leakage.

   Imputação com strategy="constant" é stateless (fill_value fixo no YAML) —
   sem risco de data leakage. Os 11 NaN de TotalCharges têm tenure=0, portanto
   TotalCharges=0 é a regra de negócio correta (sem histórico → sem cobrança).
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.pipeline import Pipeline

from src.preprocessing.transformers import (
    TypeCastTransformer,
    BinaryFlagTransformer,
    BinaryEncodingTransformer,
    TernaryEncodingTransformer,
    CategoricalEncoder,
    RatioFeatureTransformer,
    LogTransformer,
    FeatureSelector,
    GroupMedianImputer,
    ConstantImputer,
)


class PreprocessingPipelineBuilder:
    """
    Constrói um sklearn.Pipeline de feature engineering a partir do config YAML.

    Uso:
        builder = PreprocessingPipelineBuilder(config=preprocessing_cfg, logger=logger)
        pipeline = builder.build()
        df_transformado = pipeline.fit_transform(df)
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger

    def build(self) -> Pipeline:
        """
        Monta e retorna o sklearn.Pipeline com todas as etapas configuradas.

        Returns:
            sklearn.Pipeline pronto para fit_transform().

        Raises:
            KeyError: Se uma seção obrigatória estiver ausente no config.
        """
        imputation_specs = self.config.get("imputation", [])
        imputacao_steps = []
        for spec in imputation_specs:
            strategy = spec.get("strategy", "median")
            if strategy == "constant":
                transformer = ConstantImputer(
                    target_col=spec["column"],
                    fill_value=spec.get("fill_value", 0),
                    logger=self.logger,
                )
            else:
                transformer = GroupMedianImputer(
                    target_col=spec["column"],
                    group_col=spec.get("group_by"),
                    logger=self.logger,
                )
            imputacao_steps.append((f"imputacao_{spec['column']}", transformer))

        etapas = [
            ("type_cast", TypeCastTransformer(
                casts=self.config.get("type_cast", []),
                logger=self.logger,
            )),
            ("flags_binarias", BinaryFlagTransformer(
                flags=self.config.get("binary_flags", []),
                logger=self.logger,
            )),
            *imputacao_steps,
            ("encoding_binario", BinaryEncodingTransformer(
                config=self.config.get("binary_encoding", {}),
                logger=self.logger,
            )),
            ("encoding_ternario", TernaryEncodingTransformer(
                config=self.config.get("ternary_encoding", {}),
                logger=self.logger,
            )),
            ("encoding_categorico", CategoricalEncoder(
                encodings=self.config.get("categorical_encoding", []),
                logger=self.logger,
            )),
            ("razoes", RatioFeatureTransformer(
                ratios=self.config.get("ratio_features", []),
                logger=self.logger,
            )),
            ("log", LogTransformer(
                columns=self.config.get("log_transform", {}).get("columns", []),
                logger=self.logger,
            )),
            ("selecao", FeatureSelector(
                features_to_keep=self.config.get("feature_selection", {}).get("features_to_keep", []),
                logger=self.logger,
            )),
        ]

        if self.logger:
            self.logger.info(
                "PreprocessingPipelineBuilder: pipeline montado com %d etapas: %s",
                len(etapas),
                [nome for nome, _ in etapas],
            )

        return Pipeline(etapas)
