# %%
#Configuração do Ambiente
import sys
import os
from pathlib import Path 

# %%
#Definições
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
SECRETS_PATH = ROOT_DIR / 'secrets.env'
DATA_CONFIG = CONFIG_DIR / 'data.yaml'
PIPELINE_CONFIG = CONFIG_DIR / 'pipeline.yaml'
PATHS_LIST = [ROOT_DIR, CONFIG_DIR]

# %%
for _p in PATHS_LIST:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# %%
from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.downloader import check_kaggle_credentials

# %%
# fazer a leitura dos arquivos de configuração
data_cfg = load_yaml(DATA_CONFIG)
pipeline_cfg = load_yaml(PIPELINE_CONFIG)

# %%
# obtendo configuração do log
log_cfg = pipeline_cfg.get("logging")

# %%
logger = get_logger(
    name='ingestao',
    logging_config=log_cfg
)

# %%
#checando variáveis de ambiente
if check_kaggle_credentials(secrets_path=SECRETS_PATH):
    logger.info("Kaggle credentials set.")
else:
    logger.error("Kaggle credentials not set.")
    
# %%
