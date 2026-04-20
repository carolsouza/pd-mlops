# MLOps — Predição de Churn Telco

Projeto de MLOps para predição de churn de clientes de uma operadora de telecomunicações.  
Dataset: [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7.043 clientes, 33 features, 26.5% churn.

**Melhor resultado:** VotingClassifier (LR×4 + XGB×3 + GradBoost×1) — CV AUC 0.8490 | Holdout AUC 0.8558

---

## Pré-requisitos

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Credenciais Kaggle em `secrets.env` (para download automático):
```
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=sua_chave
```

---

## Pipeline — Execução em Etapas

Execute os notebooks na ordem abaixo. Cada etapa gera o arquivo de entrada da próxima.

### 1. Ingestão (`notebooks/ingestao.py`)
Baixa o dataset do Kaggle e converte para Parquet comprimido.

```
Saída: data/raw/ + data/processed/telco_customer_churn.parquet
```

> Só precisa rodar uma vez. Se já tiver o CSV, pode pular para a etapa 2.

---

### 2. Qualidade (`notebooks/qualidade.py`)
Valida o dataset com Great Expectations: schema, nulos, distribuições, faixas de valores.

```
Entrada: data/processed/telco_customer_churn.parquet
Saída:   outputs/quality/quality_report.json
```

> Configurar critérios em `config/quality.yaml`.

---

### 3. Pré-processamento (`notebooks/preprocessamento.py`)
Executa o pipeline completo de feature engineering: encoding, features derivadas, transformações log, features de razão e seleção final.

```
Entrada: data/processed/telco_customer_churn.parquet
Saída:   data/features/telco_customer_churn_features.parquet  (33 features + target)
```

> Para adicionar/remover features ou mudar encodings, edite apenas `config/preprocessing.yaml`.

---

### 4. Modelagem (`notebooks/modelagem.py`)
Treina todos os modelos habilitados, otimiza hiperparâmetros via Optuna, constrói ensembles e registra o melhor modelo no MLflow Model Registry.

```
Entrada: data/features/telco_customer_churn_features.parquet
Saída:   mlruns.db  (experimentos e modelo registrado)
         outputs/modeling/experiment_summary.json
         outputs/modeling/*.png  (plots diagnósticos)
```

> Para habilitar/desabilitar modelos, ajustar trials ou search spaces, edite `config/modeling.yaml`.  
> Um novo modelo **só é promovido** ao registry se superar o CV AUC da versão atual.

---

## Inspecionar Experimentos (MLflow UI)

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Acesse `http://localhost:5000` para ver runs, métricas por fold, artefatos e versões do modelo.

---

## Aplicação de Produção (Streamlit)

```bash
streamlit run production_app/app.py
```

**Página 1 — Predição Individual:** formulário com os dados do cliente → probabilidade de churn + nível de risco (Baixo / Médio / Alto).

**Página 2 — Monitoramento em Lote:** amostra N clientes do parquet de features, calcula métricas por lote (AUC, F1, Recall, Precision) com média móvel e detecção visual de drift.

---

## Estrutura do Projeto

```
mlops-churn/
├── config/
│   ├── preprocessing.yaml   ← política de feature engineering
│   ├── modeling.yaml        ← modelos, search spaces, métricas
│   └── quality.yaml         ← critérios de validação de dados
├── notebooks/
│   ├── ingestao.py          ← etapa 1: download + parquet
│   ├── qualidade.py         ← etapa 2: validação Great Expectations
│   ├── preprocessamento.py  ← etapa 3: feature engineering
│   └── modelagem.py         ← etapa 4: treino + MLflow + registry
├── src/
│   ├── preprocessing/       ← transformadores sklearn-compatíveis
│   └── modeling/            ← otimizador, tracker, avaliador, ensembles
├── production_app/
│   ├── app.py               ← ponto de entrada Streamlit
│   └── pages/
│       ├── 1_Predicao.py
│       └── 2_Monitoramento.py
├── data/
│   ├── processed/           ← parquet bruto pós-ingestão
│   └── features/            ← parquet com features engenheiradas
├── outputs/modeling/        ← plots, experiment_summary.json
└── mlruns.db                ← backend SQLite do MLflow
```
