"""
Módulo de configurações para o projeto MSR-TCN.
Define os segmentos do mercado financeiro, seus respectivos ativos (tickers) e os caminhos globais.
"""
import os
from typing import Dict, List

# Define os 8 segmentos do mercado financeiro utilizados no artigo para avaliação.
# Estes segmentos cobrem ações brasileiras (BlueChips, SmallCaps), Fundos Imobiliários (FIIs), 
# BDRs, Commodities, Mega-Caps dos EUA, Empresas Tradicionais dos EUA e Câmbio.
SEGMENTOS: Dict[str, List[str]] = {
    'BlueChips': [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 
        'WEGE3.SA', 'SUZB3.SA', 'RENT3.SA', 'EQTL3.SA', 'RADL3.SA'
    ],
    'SmallCaps': [
        'MGLU3.SA', 'YDUQ3.SA', 'POMO4.SA', 'TEND3.SA', 
        'TASA4.SA', 'FLRY3.SA', 'MRVE3.SA', 'RAPT4.SA'
    ],
    'FIIs': [
        'KNRI11.SA', 'HGLG11.SA', 'MXRF11.SA', 'HGBS11.SA', 'BRCR11.SA',
        'SHPH11.SA', 'HTMX11.SA', 'HGRE11.SA', 'VRTA11.SA', 'KNCR11.SA'
    ],
    'BDRs': [
        'AAPL34.SA', 'MSFT34.SA', 'AMZO34.SA', 'GOGL34.SA', 
        'DISB34.SA', 'TSLA34.SA', 'JNJB34.SA', 'M1TA34.SA'
    ],
    'CommoditiesExp': [
        'GC=F', 'SI=F', 'CL=F', 'BZ=F', 'ZS=F', 
        'ZC=F', 'KC=F', 'LE=F', 'SB=F', 'HG=F',
        'PL=F', 'PA=F', 'CC=F', 'CT=F', 'NG=F'
    ],
    'MegaCapsTech': [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 
        'META', 'TSLA', 'NFLX', 'ADBE', 'CRM', 
        'AMD', 'INTC', 'CSCO', 'QCOM', 'TXN', 
        'AVGO', 'AMAT', 'MU', 'LRCX', 'INTU'
    ],
    'TradicionaisGlobais': [
        'JNJ', 'PG', 'XOM', 'CVX', 'JPM', 
        'V', 'KO', 'PEP', 'WMT', 'MCD', 
        'NKE', 'DIS', 'BA', 'PFE', 'UNH', 
        'HD', 'VZ', 'T', 'MRK', 'ABT'
    ],
    'CambioGlobal': [
        'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 
        'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'BRL=X'
    ]
}

# Caminhos absolutos para garantir que a execução seja consistente independente do diretório de trabalho.
DIRETORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIRETORIO_DADOS = os.path.join(DIRETORIO_RAIZ, 'data')
DIRETORIO_CONFIGS = os.path.join(DIRETORIO_RAIZ, 'configs')
DIRETORIO_RESULTADOS = os.path.join(DIRETORIO_RAIZ, 'results')
DIRETORIO_CHECKPOINTS = os.path.join(DIRETORIO_RAIZ, 'checkpoints')

# Garante que os diretórios existam
for diretorio in [DIRETORIO_DADOS, DIRETORIO_CONFIGS, DIRETORIO_RESULTADOS, DIRETORIO_CHECKPOINTS]:
    os.makedirs(diretorio, exist_ok=True)
