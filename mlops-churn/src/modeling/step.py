"""
modeling/step.py — Etapa de Modelagem do Pipeline MLOps.

Implementa PipelineStep para orquestrar toda a experimentação:
  Entrada : data/features/telco_customer_churn_features.parquet (gerado por preprocessamento.py)
  Saida   : mlruns.db (tracking SQLite), outputs/modeling/ (artefatos)

Fluxo:
  1. Carrega features do Parquet
  2. Divide treino / holdout (estratificado pelo target Churn 0/1)
  3. Baseline CV — cada modelo habilitado com parametros padrao
  4. Otimizacao — OptimizerFactory seleciona estrategia do YAML (optuna|grid|random)
  5. Ensembles — Stacking e Voting sobre os top-N modelos
  6. Selecao do melhor modelo (maior CV ROC-AUC)
  7. Analise diagnostica — plots de classificacao + metricas de treino
  8. Avaliacao holdout — metricas finais em dados nunca vistos
  9. Registro no MLflow Model Registry

Toda politica (quais modelos, search_space, n_trials, metricas) vem de
config/modeling.yaml. Este arquivo NAO contem logica de negocio.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import optuna
from sklearn.model_selection import train_test_split

from src.core.base import PipelineStep
from src.utils.config_loader import load_yaml
from src.modeling.metrics import calcular_metricas, agregar_metricas_folds
from src.modeling.model_factory import construir_pipeline
from src.modeling.cross_validation import CVRunner
from src.modeling.optimizer import OptimizerFactory, _params_reducer_padrao
from src.modeling.ensemble import EnsembleBuilder
from src.modeling.evaluator import HoldoutEvaluator
from src.modeling.artifacts import ArtifactGenerator
from src.modeling.tracker import MLflowTracker

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelingStep(PipelineStep):
    """
    Etapa de experimentacao e modelagem do pipeline MLOps.

    Toda configuracao vem de modeling.yaml + pipeline.yaml — o step e
    agnostico ao dominio (funciona para qualquer dataset tabular de classificacao).

    Parametros
    ----------
    context : PipelineContext — fornece root_dir, config_dir e logger
    """

    def __init__(self, context: Any) -> None:
        super().__init__(logger=context.logger)
        self.context = context
        self._cfg    = self._carregar_config()

    def _carregar_config(self) -> dict:
        """Mescla pipeline.yaml e modeling.yaml em um unico dicionario de config."""
        pipeline_cfg  = self.context.pipeline_cfg
        modeling_cfg  = load_yaml(self.context.config_dir / 'modeling.yaml')
        return {**pipeline_cfg, **modeling_cfg}

    # ── Propriedades de configuracao ─────────────────────────────────────────

    @property
    def _modeling_cfg(self) -> dict:
        return self._cfg.get('modeling', {})

    @property
    def _seed(self) -> int:
        return self._modeling_cfg.get('random_seed', 42)

    @property
    def _pipe_cfg(self) -> dict:
        return self._cfg.get('pipeline', {})

    @property
    def _feat_red_cfg(self) -> dict:
        return self._cfg.get('feature_reduction', {})

    @property
    def _cv_cfg(self) -> dict:
        return self._cfg.get('cv', {})

    @property
    def _holdout_cfg(self) -> dict:
        return self._cfg.get('holdout', {})

    @property
    def _models_cfg(self) -> dict:
        return self._cfg.get('models', {})

    @property
    def _ensembles_cfg(self) -> dict:
        return self._cfg.get('ensembles', {})

    @property
    def _artifacts_cfg(self) -> dict:
        return self._cfg.get('artifacts', {})

    @property
    def _output_dir(self) -> Path:
        return self.context.root_dir / self._artifacts_cfg.get('output_dir', 'outputs/modeling')

    @property
    def _n_trials_global(self) -> int:
        opt_cfg = self._cfg.get('optimizer', {})
        return opt_cfg.get('optuna', {}).get('default_trials', 10)

    # ── Passo principal ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Executa a etapa completa de modelagem."""
        self.logger.info('=== Modelagem e Experimentacao — MLOps Pipeline ===')

        # ── 1. Carrega features ───────────────────────────────────────────────
        X, y = self._carregar_features()

        # ── 2. Divide treino / holdout ────────────────────────────────────────
        X_train, X_holdout, y_train, y_holdout = self._dividir_treino_holdout(X, y)

        # ── 3. Instancia infraestrutura ───────────────────────────────────────
        cv_runner = CVRunner.de_config(self._cv_cfg, self._seed)
        tracker   = MLflowTracker(
            tracking_uri   =self._modeling_cfg.get('tracking_uri', 'sqlite:///mlruns.db'),
            experiment_name=self._modeling_cfg.get('experiment_name', 'telco-churn-experiments'),
            root_dir       =self.context.root_dir,
            logger         =self.logger,
        )
        otimizador = OptimizerFactory.criar(
            cfg_modelagem=self._cfg,
            cv_runner    =cv_runner,
            pipe_cfg     =self._pipe_cfg,
            seed         =self._seed,
            logger       =self.logger,
        )
        gerador_artefatos = ArtifactGenerator(
            output_dir   =self._output_dir,
            artifacts_cfg=self._artifacts_cfg,
            logger       =self.logger,
        )
        avaliador = HoldoutEvaluator(logger=self.logger)

        # ── 4. Baseline CV ────────────────────────────────────────────────────
        todos_resultados = self._executar_baseline(
            cv_runner, tracker, X_train, y_train
        )

        # ── 5. Otimizacao de hiperparametros ──────────────────────────────────
        todos_resultados = self._executar_otimizacao(
            otimizador, tracker, cv_runner, todos_resultados, X_train, y_train
        )

        # ── 6. Ensembles ──────────────────────────────────────────────────────
        todos_resultados = self._executar_ensembles(
            todos_resultados, cv_runner, X_train, y_train
        )

        # ── 7. Selecao do melhor modelo ───────────────────────────────────────
        nome_melhor, resultado_melhor = self._selecionar_melhor(todos_resultados)

        # ── 8. Analise diagnostica ────────────────────────────────────────────
        melhor_modelo = self._treinar_melhor_modelo(nome_melhor, resultado_melhor)
        melhor_modelo.fit(X_train, y_train)

        plot_paths, metricas_treino = gerador_artefatos.gerar_diagnosticos_modelo(
            model       =melhor_modelo,
            model_name  =nome_melhor,
            X_train     =X_train,
            y_train     =y_train,
            fold_metrics=resultado_melhor['fold_metrics'],
        )

        # ── 9. Loga melhor modelo ─────────────────────────────────────────────
        best_run_id = tracker.logar_melhor_modelo(
            model_name    =nome_melhor,
            model         =melhor_modelo,
            best_params   =resultado_melhor['best_params'],
            reducer_params=resultado_melhor.get('reducer_params', _params_reducer_padrao(self._feat_red_cfg)),
            cv_metrics    ={
                'cv_roc_auc_mean': resultado_melhor['cv_roc_auc_mean'],
                'cv_roc_auc_std' : resultado_melhor['cv_roc_auc_std'],
                'cv_f1_mean'     : resultado_melhor['cv_f1_mean'],
                'cv_recall_mean' : resultado_melhor['cv_recall_mean'],
                'cv_precision_mean': resultado_melhor['cv_precision_mean'],
            },
            train_metrics =metricas_treino,
            fold_metrics  =resultado_melhor['fold_metrics'],
            plot_paths    =plot_paths,
            tuned         =resultado_melhor['tuned'],
        )

        # ── 10. Avaliacao holdout ─────────────────────────────────────────────
        metricas_holdout = avaliador.avaliar(melhor_modelo, X_holdout, y_holdout)
        delta_pp = resultado_melhor['cv_roc_auc_mean'] - metricas_holdout['roc_auc']
        avaliador.diagnosticar_robustez(resultado_melhor['cv_roc_auc_mean'], metricas_holdout['roc_auc'])

        y_prob_holdout = (
            melhor_modelo.predict_proba(X_holdout)[:, 1]
            if hasattr(melhor_modelo, 'predict_proba') else None
        )
        plot_holdout = gerador_artefatos.plot_holdout_evaluation(
            y_holdout      =y_holdout,
            y_pred_holdout =melhor_modelo.predict(X_holdout),
            holdout_metrics=metricas_holdout,
            model_name     =nome_melhor,
            y_prob_holdout =y_prob_holdout,
        )
        tracker.logar_holdout(
            run_id           =best_run_id,
            holdout_metrics  =metricas_holdout,
            delta_pct        =delta_pp,
            holdout_plot_path=plot_holdout,
        )

        # ── 11. Registro no Model Registry ───────────────────────────────────
        registry_name = self._modeling_cfg.get('registry_name', 'telco-churn-best')
        tracker.registrar_modelo(best_run_id, registry_name)

        # ── 12. Resumo JSON ───────────────────────────────────────────────────
        ranking_registros = [
            {
                'modelo'          : k,
                'cv_roc_auc_mean' : v['cv_roc_auc_mean'],
                'cv_f1_mean'      : v['cv_f1_mean'],
            }
            for k, v in sorted(
                todos_resultados.items(),
                key=lambda x: x[1]['cv_roc_auc_mean'],
                reverse=True,
            )
        ]
        top_n = self._ensembles_cfg.get('top_n_base_models', 3)
        top_n_nomes = [nome for nome, _ in sorted(
            todos_resultados.items(), key=lambda x: x[1]['cv_roc_auc_mean'], reverse=True
        )[:top_n]]

        tracker.salvar_resumo_json(
            output_dir          =self._output_dir,
            best_model_name     =nome_melhor,
            best_run_id         =best_run_id,
            best_result         =resultado_melhor,
            holdout_metrics     =metricas_holdout,
            top_n_names         =top_n_nomes,
            full_ranking_records=ranking_registros,
        )

        self.logger.info('=' * 60)
        self.logger.info('=== Modelagem CONCLUIDA ===')
        self.logger.info('Melhor modelo     : %s', nome_melhor)
        self.logger.info('  CV AUC          : %.4f +- %.4f',
                         resultado_melhor['cv_roc_auc_mean'], resultado_melhor['cv_roc_auc_std'])
        self.logger.info('  CV F1           : %.4f', resultado_melhor['cv_f1_mean'])
        self.logger.info('  Holdout AUC     : %.4f (delta=%.4f pp)',
                         metricas_holdout['roc_auc'], delta_pp)
        self.logger.info('  Holdout F1      : %.4f', metricas_holdout['f1'])
        self.logger.info('  Holdout Recall  : %.4f', metricas_holdout['recall'])
        self.logger.info('=' * 60)

    # ── Metodos privados auxiliares ───────────────────────────────────────────

    def _carregar_features(self) -> tuple[pd.DataFrame, pd.Series]:
        """Carrega o Parquet de features e separa X e y."""
        features_cfg  = self._cfg.get('paths', {})
        features_dir  = self.context.root_dir / features_cfg.get('features_data_dir', 'data/features')
        features_file = features_dir / features_cfg.get(
            'features_filename', 'telco_customer_churn_features.parquet'
        )

        self.logger.info('-' * 60)
        self.logger.info('SECAO 1: Carregar Features — %s', features_file)

        if not features_file.exists():
            raise FileNotFoundError(
                f'Arquivo de features nao encontrado: {features_file}\n'
                'Execute preprocessamento.py antes deste script.'
            )

        df = pq.read_table(str(features_file)).to_pandas()
        self.logger.info('Shape: %s', df.shape)

        sel_cfg    = self._cfg.get('feature_selection', {})
        target_col = sel_cfg.get('target', 'Churn')
        feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].copy()
        y = df[target_col]

        # XGBoost rejeita nomes de colunas com '[', ']' ou '<'
        rename_map = {
            c: c.replace('<', 'lt_').replace('[', '(').replace(']', ')')
            for c in X.columns
            if any(ch in c for ch in ('<', '[', ']'))
        }
        if rename_map:
            X = X.rename(columns=rename_map)
            self.logger.info('Colunas renomeadas para XGBoost: %s', rename_map)

        n_nulos = X.isna().sum().sum()
        if n_nulos > 0:
            self.logger.warning('ATENCAO: %d valores nulos encontrados nas features!', n_nulos)
        else:
            self.logger.info('Sem valores nulos nas features')

        churn_rate = float(y.mean())
        self.logger.info(
            'Features: %d | Target: %s | Churn rate: %.1f%% (pos=%d, neg=%d)',
            len(feature_cols), target_col, churn_rate * 100,
            int(y.sum()), int((y == 0).sum()),
        )
        return X, y

    def _dividir_treino_holdout(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide os dados em treino e holdout com estratificacao pelo target."""
        n_bins    = self._holdout_cfg.get('stratify_bins', 2)
        test_size = self._holdout_cfg.get('test_size', 0.2)

        self.logger.info('-' * 60)
        self.logger.info('SECAO 2: Divisao Treino / Holdout')
        self.logger.info('Test size: %.0f%%  |  Bins de estratificacao: %d', test_size * 100, n_bins)

        # Para target binario (ou poucos valores unicos), estratifica diretamente
        # por y, evitando falhas do pd.qcut com distribuicoes muito desbalanceadas.
        if y.nunique() <= n_bins:
            y_estratificacao = y
        else:
            y_estratificacao = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=test_size, random_state=self._seed, stratify=y_estratificacao,
        )

        self.logger.info('Treino  : %d amostras (%.1f%%)', len(X_train), 100 * len(X_train) / len(X))
        self.logger.info('Holdout : %d amostras (%.1f%%)', len(X_holdout), 100 * len(X_holdout) / len(X))
        return X_train, X_holdout, y_train, y_holdout

    def _executar_baseline(
        self,
        cv_runner: CVRunner,
        tracker: MLflowTracker,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict:
        """Executa CV de baseline para todos os modelos habilitados."""
        self.logger.info('-' * 60)
        self.logger.info('SECAO 3: Baseline CV (%d folds)', self._cv_cfg.get('n_splits', 5))

        todos_resultados: dict = {}
        params_reducer_padrao = _params_reducer_padrao(self._feat_red_cfg)

        for nome_modelo, cfg_modelo in self._models_cfg.items():
            if not cfg_modelo.get('enabled', True):
                self.logger.info('  [SKIP] %s (desabilitado)', nome_modelo)
                continue

            self.logger.info('  [BASELINE] %-25s ...', nome_modelo)
            pipeline = construir_pipeline(
                model_cfg     =cfg_modelo,
                params_modelo =None,
                params_reducer=params_reducer_padrao,
                pipe_cfg      =self._pipe_cfg,
            )
            t0 = time.time()
            fold_mets = cv_runner.executar(pipeline, X_train, y_train)
            agg       = agregar_metricas_folds(fold_mets)
            tempo     = time.time() - t0

            tracker.logar_baseline(
                model_name   =nome_modelo,
                params       =dict(cfg_modelo.get('default_params') or {}),
                fold_metrics =fold_mets,
                agg_metrics  =agg,
                training_time=tempo,
                model_class  =f"{cfg_modelo['module']}.{cfg_modelo['class']}",
                reducer_method=params_reducer_padrao.get('method', 'none'),
            )

            todos_resultados[nome_modelo] = {
                **agg,
                'fold_metrics'  : fold_mets,
                'model_cfg'     : cfg_modelo,
                'best_params'   : dict(cfg_modelo.get('default_params') or {}),
                'reducer_params': params_reducer_padrao,
                'tuned'         : False,
            }
            self.logger.info(
                '    CV AUC: %.4f +- %.4f  |  F1: %.4f  |  Recall: %.4f  |  %.1fs',
                agg['cv_roc_auc_mean'], agg['cv_roc_auc_std'],
                agg['cv_f1_mean'], agg['cv_recall_mean'], tempo,
            )

        return todos_resultados

    def _executar_otimizacao(
        self,
        otimizador: Any,
        tracker: MLflowTracker,
        cv_runner: CVRunner,
        todos_resultados: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict:
        """Otimiza hiperparametros de cada modelo habilitado."""
        self.logger.info('-' * 60)
        self.logger.info('SECAO 4: Otimizacao de Hiperparametros')

        for nome_modelo, cfg_modelo in self._models_cfg.items():
            if not cfg_modelo.get('enabled', True):
                continue

            search_space = cfg_modelo.get('search_space') or {}
            n_trials     = cfg_modelo.get('optuna_trials', self._n_trials_global)

            if not search_space or n_trials <= 1:
                self.logger.info('  [SKIP OPT] %-20s — sem hiperparametros', nome_modelo)
                continue

            self.logger.info('  [OPT] %-22s  (%d trials) ...', nome_modelo, n_trials)
            t0 = time.time()

            # Subsampling para modelos lentos (ex: SVC O(n²))
            max_s = cfg_modelo.get('max_samples_for_tuning')
            if max_s and len(X_train) > max_s:
                rng      = np.random.default_rng(self._seed)
                idx_tune = rng.choice(len(X_train), max_s, replace=False)
                X_tune   = X_train.iloc[idx_tune]
                y_tune   = y_train.iloc[idx_tune]
                self.logger.info('    Subsampling: %d -> %d amostras', len(X_train), max_s)
            else:
                X_tune = X_train
                y_tune = y_train

            with tracker.contexto_otimizacao(nome_modelo):
                resultado_opt = otimizador.otimizar(
                    model_name  =nome_modelo,
                    model_cfg   =cfg_modelo,
                    X_tune      =X_tune,
                    y_tune      =y_tune,
                    pipe_cfg    =self._pipe_cfg,
                    feat_red_cfg=self._feat_red_cfg,
                )

                if not resultado_opt:
                    continue

                # Loga resultado final do otimizador no run pai
                study = resultado_opt.get('study')
                if study:
                    tracker.logar_melhor_optuna(
                        best_params     =study.best_params,
                        best_cv_roc_auc =study.best_value,
                        n_trials        =len(study.trials),
                        study           =study,
                        artifact_paths  =[
                            self._output_dir / f'optuna_history_{nome_modelo}.png',
                            self._output_dir / f'optuna_params_{nome_modelo}.png',
                        ],
                    )
                    artifact_gen = ArtifactGenerator(
                        output_dir   =self._output_dir,
                        artifacts_cfg=self._artifacts_cfg,
                        logger       =self.logger,
                    )
                    artifact_gen.plot_optuna_history(study, nome_modelo)
                    artifact_gen.plot_optuna_param_importances(study, nome_modelo)

            # Avalia com os melhores parametros encontrados no X_train completo
            params_estimador = resultado_opt.get('estimator_params', {})
            params_reducer   = resultado_opt.get('reducer_params', _params_reducer_padrao(self._feat_red_cfg))

            pipeline_otimizado = construir_pipeline(
                model_cfg     =cfg_modelo,
                params_modelo =params_estimador,
                params_reducer=params_reducer,
                pipe_cfg      =self._pipe_cfg,
            )
            fold_mets_otimizado = cv_runner.executar(pipeline_otimizado, X_train, y_train)
            agg_otimizado       = agregar_metricas_folds(fold_mets_otimizado)

            auc_anterior = todos_resultados[nome_modelo]['cv_roc_auc_mean']
            if agg_otimizado['cv_roc_auc_mean'] > auc_anterior:
                todos_resultados[nome_modelo].update({
                    **agg_otimizado,
                    'fold_metrics'  : fold_mets_otimizado,
                    'best_params'   : params_estimador,
                    'reducer_params': params_reducer,
                    'tuned'         : True,
                })
                self.logger.info(
                    '    Melhoria: %.4f -> %.4f (delta=+%.4f)  %.1fs',
                    auc_anterior, agg_otimizado['cv_roc_auc_mean'],
                    agg_otimizado['cv_roc_auc_mean'] - auc_anterior,
                    time.time() - t0,
                )
            else:
                self.logger.info(
                    '    Sem melhoria: baseline %.4f mantido  %.1fs',
                    auc_anterior, time.time() - t0,
                )

        return todos_resultados

    def _executar_ensembles(
        self,
        todos_resultados: dict,
        cv_runner: CVRunner,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict:
        """Constroi Stacking e Voting com os top-N modelos individuais."""
        self.logger.info('-' * 60)
        self.logger.info('SECAO 5: Ensembles')

        top_n = self._ensembles_cfg.get('top_n_base_models', 3)
        top_n_entries = sorted(
            todos_resultados.items(), key=lambda x: x[1]['cv_roc_auc_mean'], reverse=True
        )[:top_n]
        nomes_base = [nome for nome, _ in top_n_entries]
        self.logger.info('Top-%d modelos base: %s', top_n, nomes_base)

        builder = EnsembleBuilder(
            ensembles_cfg  =self._ensembles_cfg,
            cv_runner      =cv_runner,
            pipe_cfg       =self._pipe_cfg,
            feat_red_cfg   =self._feat_red_cfg,
            n_trials_global=self._n_trials_global,
            seed           =self._seed,
            logger         =self.logger,
        )

        resultado_stacking = builder.construir_stacking(top_n_entries, X_train, y_train)
        if resultado_stacking:
            todos_resultados['stacking'] = resultado_stacking

        resultado_voting = builder.construir_voting(top_n_entries, X_train, y_train)
        if resultado_voting:
            todos_resultados['voting'] = resultado_voting

        return todos_resultados

    def _selecionar_melhor(self, todos_resultados: dict) -> tuple[str, dict]:
        """Seleciona o modelo com maior CV ROC-AUC medio."""
        self.logger.info('-' * 60)
        self.logger.info('SECAO 6: Selecao do Melhor Modelo')

        ranking = sorted(
            todos_resultados.items(),
            key=lambda x: (x[1]['cv_roc_auc_mean'], -x[1]['cv_roc_auc_std']),
            reverse=True,
        )
        for nome, res in ranking:
            self.logger.info(
                '  %-25s CV AUC: %.4f +- %.4f  F1: %.4f  %s',
                nome, res['cv_roc_auc_mean'], res['cv_roc_auc_std'], res['cv_f1_mean'],
                'tunado' if res['tuned'] else '— baseline',
            )

        nome_melhor, resultado_melhor = ranking[0]
        self.logger.info(
            'Melhor modelo: %s  (CV AUC: %.4f)', nome_melhor, resultado_melhor['cv_roc_auc_mean']
        )
        return nome_melhor, resultado_melhor

    def _treinar_melhor_modelo(self, nome_melhor: str, resultado_melhor: dict) -> Any:
        """Instancia o melhor modelo (usa _instance pre-criada para ensembles)."""
        if '_instance' in resultado_melhor:
            return resultado_melhor['_instance']
        return construir_pipeline(
            model_cfg     =resultado_melhor['model_cfg'],
            params_modelo =resultado_melhor['best_params'],
            params_reducer=resultado_melhor.get('reducer_params', _params_reducer_padrao(self._feat_red_cfg)),
            pipe_cfg      =self._pipe_cfg,
        )
