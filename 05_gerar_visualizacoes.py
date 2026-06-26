"""
Script de Geração de Visualizações (Data Storytelling).
Produz gráficos de alta qualidade para explicar os resultados do modelo MSR-TCN,
incluindo Matrizes de Confusão, Boxplots de retornos diários por predição, 
e as curvas de crescimento de capital acumulado.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from configuracoes import DIRETORIO_RESULTADOS

DIRETORIO_STORYTELLING = os.path.join(DIRETORIO_RESULTADOS, 'data_storytelling')

def configurar_estilo_storytelling():
    plt.style.use('seaborn-v0_8-white')
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']

def plotar_matriz_confusao(alvos, predicoes, nome_segmento, caminho_salvar):
    cm = confusion_matrix(alvos, predicoes, labels=[0, 1, 2])
    # Calcula proporção por linha (Rótulo Real)
    soma = cm.sum(axis=1, keepdims=True)
    soma[soma == 0] = 1 # Evita divisão por zero
    cm_pct = cm.astype('float') / soma * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['MANTER (0)', 'VENDER (1)', 'COMPRAR (2)'], 
                yticklabels=['MANTER (0)', 'VENDER (1)', 'COMPRAR (2)'],
                cbar_kws={'label': 'Proporção (%)'},
                annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
                
    ax.set_xlabel('Previsão do MSR-TCN', fontweight='bold')
    ax.set_ylabel('Rótulo Real (Tendência)', fontweight='bold')
    ax.set_title(f'Matriz de Confusão MSR-TCN: {nome_segmento}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(caminho_salvar, dpi=300)
    plt.close()

def plotar_boxplots_predicao(df, nome_segmento, diretorio_saida):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Mapear cores: HOLD (Cinza), SELL (Vermelho), BUY (Verde)
    sns.boxplot(x='MSRTCN_Pred', y='Ret_1d', data=df, 
                   palette=['#9E9E9E', '#D32F2F', '#388E3C'], ax=ax)
    
    ax.set_xticklabels(['MANTER (0)', 'VENDER (1)', 'COMPRAR (2)'])
    ax.axhline(0, color='#333333', linestyle='--', linewidth=1.5)
    ax.set_title(f'Retornos Diários Reais por Predição: {nome_segmento}', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Retorno Real de 1 Dia (%)', fontweight='bold')
    ax.set_xlabel('Decisão Algorítmica do MSR-TCN', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_saida, f'boxplots_retornos_{nome_segmento}.png'), dpi=300)
    plt.close()

def plotar_retornos_acumulados_storytelling(ticker, nome_segmento, df_ticker, diretorio_saida):
    def calcular_retornos_estrategia(predicoes, retornos_1d):
        n = len(predicoes)
        retornos_est = np.zeros(n)
        posicao = 0.0  
        
        predicoes_arr = np.array(predicoes)
        ret_arr = np.array(retornos_1d)
        
        for i in range(n):
            if predicoes_arr[i] == 2: # COMPRAR
                posicao = 1.0
            elif predicoes_arr[i] == 1: # VENDER
                posicao = 0.0
            
            retornos_est[i] = posicao * (ret_arr[i] / 100.0)
            
        return np.cumprod(1 + retornos_est)

    df_ticker = df_ticker.sort_values('Data').reset_index(drop=True)
    datas = pd.to_datetime(df_ticker['Data'])
    
    retornos_bnh = np.cumprod(1 + (df_ticker['Ret_1d'].values / 100.0))
    est_msr = calcular_retornos_estrategia(df_ticker['MSRTCN_Pred'].values, df_ticker['Ret_1d'].values)
    est_base = calcular_retornos_estrategia(df_ticker['BaselineTCN_Pred'].values, df_ticker['Ret_1d'].values)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(datas, retornos_bnh, label='Mercado (Buy & Hold)', color='#B0BEC5', linestyle='--', linewidth=2)
    ax.plot(datas, est_base, label='Estratégia TCN Controle', color='#F24236', linewidth=2, alpha=0.8)
    ax.plot(datas, est_msr, label='Estratégia MSR-TCN', color='#2E86AB', linewidth=3)
    
    ax.set_title(f'Crescimento de Capital: {ticker} (Segmento {nome_segmento})', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Multiplicador do Capital (1.0 = Inicial)', fontweight='bold')
    ax.legend(frameon=False, fontsize=12)
    
    idx_max = np.argmax(est_msr)
    ax.annotate(f'Pico MSR-TCN: {est_msr[idx_max]:.2f}x', 
                xy=(datas.iloc[idx_max], est_msr[idx_max]),
                xytext=(datas.iloc[idx_max], est_msr[idx_max] * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_saida, f'evolucao_capital_{ticker}.png'), dpi=300)
    plt.close()

def main():
    configurar_estilo_storytelling()
    os.makedirs(DIRETORIO_STORYTELLING, exist_ok=True)
    print("Gerando Data Storytelling (Segmentos Vencedores)...")

    segmentos_vencedores = ['SmallCaps', 'FIIs', 'MegaCapsTech', 'TradicionaisGlobais', 'CambioGlobal']
    
    for segmento in segmentos_vencedores:
        caminho_csv = os.path.join(DIRETORIO_RESULTADOS, f"predicoes_{segmento}_wf.csv")
        if not os.path.exists(caminho_csv):
            print(f"  ⚠ CSV não encontrado para {segmento}. Aguardando treinamento.")
            continue
            
        print(f"Processando Segmento: {segmento}...")
        df = pd.read_csv(caminho_csv)
        
        plotar_matriz_confusao(df['Alvo'].values, df['MSRTCN_Pred'].values, segmento, os.path.join(DIRETORIO_STORYTELLING, f"matriz_confusao_{segmento}.png"))
        plotar_boxplots_predicao(df, segmento, DIRETORIO_STORYTELLING)
        
        # Seleciona 1 ativo representativo para contar a história
        ativos = df['Ativo'].unique()
        if len(ativos) > 0:
            representativo = ativos[0]
            if segmento == 'SmallCaps' and 'YDUQ3.SA' in ativos: representativo = 'YDUQ3.SA'
            if segmento == 'FIIs' and 'MXRF11.SA' in ativos: representativo = 'MXRF11.SA'
            if segmento == 'MegaCapsTech' and 'AAPL' in ativos: representativo = 'AAPL'
            if segmento == 'TradicionaisGlobais' and 'JNJ' in ativos: representativo = 'JNJ'
            if segmento == 'CambioGlobal' and 'EURUSD=X' in ativos: representativo = 'EURUSD=X'
            
            df_ativo = df[df['Ativo'] == representativo].copy()
            plotar_retornos_acumulados_storytelling(representativo, segmento, df_ativo, DIRETORIO_STORYTELLING)

    print(f"\n✅ Relatório de Visualizações (Data Storytelling) concluído com sucesso!")
    print(f"Confira a pasta: {DIRETORIO_STORYTELLING}")

if __name__ == '__main__':
    main()
