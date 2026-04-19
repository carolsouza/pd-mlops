# %%
# ─────────────────────────────────────────────────────────────────────────────
# MLOps — Pré-processamento e Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
#
# TERCEIRA etapa do pipeline de dados.
#   Entrada : data/processed/telco_customer_churn.parquet  ← gerado por qualidade.py
#   Saída   : data/features/telco_customer_churn_features.parquet
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/preprocessing.yaml  (O QUÊ transformar e parâmetros)
#   • Mecanismo → src/preprocessing/         (COMO executar cada transformação)
#
# Para ajustar qualquer transformação (nova feature, novo encoding, etc.),
# edite apenas o YAML. O código em src/preprocessing/ não precisa mudar.
# ─────────────────────────────────────────────────────────────────────────────

# %%
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "config")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.core.context import PipelineContext
from src.preprocessing import PreprocessingStep

# %%
ctx  = PipelineContext.from_notebook(__file__)
step = PreprocessingStep(ctx)
step.run()