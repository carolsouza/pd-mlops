"""
preprocessing/ — Módulo de Pré-processamento e Feature Engineering.

API pública:
  • Transformadores stateless (seguros para aplicar antes do split):
      TypeCastTransformer, BinaryFlagTransformer,
      BinaryEncodingTransformer, TernaryEncodingTransformer,
      CategoricalEncoder, RatioFeatureTransformer, LogTransformer, FeatureSelector

  • Transformadores stateful (use APENAS no pipeline de modelagem, após o split):
      GroupMedianImputer, StandardScalerTransformer

  • Infraestrutura:
      BaseFeatureTransformer  — classe base abstrata
      PreprocessingPipelineBuilder — monta sklearn.Pipeline a partir do config
      PreprocessingStep        — etapa PipelineStep completa (carrega, transforma, persiste)
"""
from src.preprocessing.base import BaseFeatureTransformer
from src.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from src.preprocessing.step import PreprocessingStep
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
    StandardScalerTransformer,
)

__all__ = [
    # Infraestrutura
    "BaseFeatureTransformer",
    "PreprocessingPipelineBuilder",
    "PreprocessingStep",
    # Transformadores stateless — pipeline do churn
    "TypeCastTransformer",
    "BinaryFlagTransformer",
    "BinaryEncodingTransformer",
    "TernaryEncodingTransformer",
    "CategoricalEncoder",
    "RatioFeatureTransformer",
    "LogTransformer",
    "FeatureSelector",
    # Stateful (somente pipeline de modelagem)
    "GroupMedianImputer",
    "StandardScalerTransformer",
]
