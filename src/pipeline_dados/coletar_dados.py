"""
Pipeline de extração de dados e engenharia de características (feature engineering) para o MSR-TCN.
Baixa dados históricos via yfinance, calcula indicadores técnicos
e gera os rótulos (labels) com base em uma abordagem de janela deslizante centralizada.
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

# Garante a importação a partir de src.configuracoes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from configuracoes import SEGMENTOS, DIRETORIO_DADOS

def processar_ativo(ticker: str):
    """
    Baixa dados históricos para um determinado ativo (ticker), calcula indicadores técnicos
    e aplica o algoritmo de rotulagem de janela deslizante centralizada (topos e fundos).
    """
    print(f"Baixando dados para {ticker}...")
    df = yf.download(ticker, start='2009-01-01', end='2024-12-20', progress=False)
    
    if df.empty:
        print(f"Falha ao baixar os dados para {ticker}.")
        return
        
    # Achata as colunas MultiIndex, se o yfinance as retornar dessa forma
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # 1. Engenharia de Características (Feature Engineering)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    # 2. Rotulagem de Topo/Fundo (janela centralizada de 11 dias)
    window = 11
    
    # Inicializa o Rótulo (Label) como 0 (Manter/Hold)
    df['Label'] = 0
    
    # Calcula os mínimos e máximos contínuos para a janela centralizada
    # min_periods=window garante que não rotulemos as bordas com janelas incompletas
    rolling_min = df['Close'].rolling(window=window, center=True, min_periods=window).min()
    rolling_max = df['Close'].rolling(window=window, center=True, min_periods=window).max()
    
    # Se o fechamento atual for o mínimo da janela -> 2 (COMPRAR/BUY)
    df.loc[df['Close'] == rolling_min, 'Label'] = 2
    
    # Se o fechamento atual for o máximo da janela -> 1 (VENDER/SELL)
    df.loc[df['Close'] == rolling_max, 'Label'] = 1
    
    df.dropna(inplace=True)
    
    # Os dados são salvos sem padronização (unscaled) para evitar vazamento de dados. 
    # A padronização será aplicada dinamicamente pelo Dataset do PyTorch usando apenas as estatísticas do conjunto de treino.
    nome = ticker.replace('^', '').replace('=', '_')
    caminho_saida = os.path.join(DIRETORIO_DADOS, f"{nome}_full.csv")
    df.to_csv(caminho_saida)
    print(f"Salvo {ticker} em {caminho_saida}")

def main():
    print("Iniciando o pipeline de ingestão de dados...")
    for segmento, ativos in SEGMENTOS.items():
        print(f"\nProcessando Segmento: {segmento}")
        for ativo in ativos:
            processar_ativo(ativo)
    print("\nIngestão de dados concluída com sucesso.")

if __name__ == "__main__":
    main()
