"""Exporta todos os transformadores do subpacote transformers."""
from src.preprocessing.transformers.type_cast import TypeCastTransformer
from src.preprocessing.transformers.binary_flags import BinaryFlagTransformer
from src.preprocessing.transformers.binary_encoding import BinaryEncodingTransformer
from src.preprocessing.transformers.ternary_encoding import TernaryEncodingTransformer
from src.preprocessing.transformers.categorical_encoder import CategoricalEncoder
from src.preprocessing.transformers.ratio_features import RatioFeatureTransformer
from src.preprocessing.transformers.log_transform import LogTransformer
from src.preprocessing.transformers.feature_selector import FeatureSelector
from src.preprocessing.transformers.stateful import GroupMedianImputer, StandardScalerTransformer, ConstantImputer

__all__ = [
    # Pipeline do churn (stateless)
    "TypeCastTransformer",
    "BinaryFlagTransformer",
    "BinaryEncodingTransformer",
    "TernaryEncodingTransformer",
    "CategoricalEncoder",
    "RatioFeatureTransformer",
    "LogTransformer",
    "FeatureSelector",
    # Stateful (usar apenas no pipeline de modelagem, após o split)
    "GroupMedianImputer",
    "StandardScalerTransformer",
    "ConstantImputer",
]
