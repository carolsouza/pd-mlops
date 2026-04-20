"""
production_app/app.py — Ponto de entrada da aplicação Streamlit.

Como executar:
    cd mlops-churn
    streamlit run production_app/app.py

Páginas disponíveis:
    1_Predicao.py      — Predição individual de churn com probabilidade e nível de risco
    2_Monitoramento.py — Dashboard de monitoramento por lotes com métricas de classificação
"""
import streamlit as st

st.switch_page("pages/1_Predicao.py")
