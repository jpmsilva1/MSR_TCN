"""
Script de Avaliação Estatística para o MSR-TCN.
Realiza o Teste de McNemar e testes de Bootstrap para o F1-Score, avaliando
a significância estatística da superioridade do MSR-TCN sobre a TCN de Controle (Baseline).
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from configuracoes import DIRETORIO_RESULTADOS

def configurar_estilo_visualizacao():
    """Configura o estilo visual dos gráficos (padrão acadêmico/profissional)."""
    plt.rcParams.update({
        'figure.facecolor': '#FAFAFA',
        'axes.facecolor': '#FAFAFA',
        'axes.edgecolor': '#E0E0E0',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#666666',
        'ytick.color': '#666666',
        'grid.color': '#F0F0F0',
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    sns.set_theme(style='white', context='talk')

def calcular_mcnemar_pareado(alvo, pred_a, pred_b):
    """
    Calcula o p-valor do Teste de McNemar para avaliar se a diferença de precisão
    entre dois modelos é estatisticamente significativa.
    """
    correto_a = (pred_a == alvo).values
    correto_b = (pred_b == alvo).values

    a_errou_b_acertou = sum((~correto_a) & correto_b)
    a_acertou_b_errou = sum(correto_a & (~correto_b))
    ambos_acertaram = sum(correto_a & correto_b)
    ambos_erraram = sum((~correto_a) & (~correto_b))

    tabela = [[ambos_acertaram, a_errou_b_acertou],
              [a_acertou_b_errou, ambos_erraram]]
    
    n_discordantes = a_errou_b_acertou + a_acertou_b_errou
    is_exact = n_discordantes < 25
    resultado = mcnemar(tabela, exact=is_exact, correction=True)
    return resultado.pvalue

def calcular_bootstrap_f1(alvo, pred_a, pred_b, n_iteracoes=1000):
    """
    Calcula a significância (p-valor) via Bootstrap para verificar se o Modelo B (MSR-TCN)
    é sistematicamente superior ao Modelo A (Baseline) na métrica de F1-Score.
    """
    f1_a_orig = f1_score(alvo, pred_a, average='macro')
    f1_b_orig = f1_score(alvo, pred_b, average='macro')
    
    n = len(alvo)
    alvo = np.array(alvo)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    
    contador_b_melhor = 0
    for _ in range(n_iteracoes):
        indices = np.random.choice(n, n, replace=True)
        f1_a = f1_score(alvo[indices], pred_a[indices], average='macro', zero_division=0)
        f1_b = f1_score(alvo[indices], pred_b[indices], average='macro', zero_division=0)
        if f1_b > f1_a:
            contador_b_melhor += 1
            
    p_valor = 1.0 - (contador_b_melhor / n_iteracoes)
    return p_valor, f1_a_orig, f1_b_orig

def plotar_p_valores(df_resultados, diretorio_saida):
    """
    Gera um gráfico de barras destacando o grau de certeza (-log10 do p-valor)
    da superioridade do MSR-TCN sobre a TCN de Controle em cada segmento.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    eps = 1e-300
    df_resultados['-log10(p)'] = -np.log10(df_resultados['p_value_F1'].astype(float) + eps)
    cores = ['#388E3C' if p < 0.05 else '#B0BEC5' for p in df_resultados['p_value_F1']]
    
    barras = sns.barplot(x='Segmento', y='-log10(p)', data=df_resultados, palette=cores, ax=ax)
    
    limiar = -np.log10(0.05)
    ax.axhline(limiar, color='#D32F2F', linestyle='--', label=rf'Significância Mínima ($\alpha=0.05$)', linewidth=2)
    
    ax.set_title('Significância da Evolução no F1-Score: MSR-TCN vs TCN Controle (Bootstrap)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Grau de Certeza (-log10 p-value)', fontweight='bold')
    ax.set_xlabel('Segmento Econômico', fontweight='bold')
    plt.xticks(rotation=45)
    ax.legend(frameon=False)
    
    for idx, barra in enumerate(barras.patches):
        pval = df_resultados['p_value_F1'].iloc[idx]
        if pval < 0.05:
            ax.text(barra.get_x() + barra.get_width()/2., barra.get_height() + 0.5,
                    f'p<{pval:.3f}', ha='center', va='bottom', fontsize=9, color='#388E3C', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_saida, 'significancia_f1_segmentos.png'), dpi=300)
    plt.close()

def main():
    configurar_estilo_visualizacao()
    diretorio_analises = os.path.join(DIRETORIO_RESULTADOS, 'analises_estatisticas')
    os.makedirs(diretorio_analises, exist_ok=True)
    
    # Busca os arquivos de predições gerados pela validação Walk-Forward
    arquivos_predicoes = glob.glob(os.path.join(DIRETORIO_RESULTADOS, 'predicoes_*_wf.csv'))
    
    if not arquivos_predicoes:
        print("Execute o script de Treinamento Walk-Forward (02_) primeiro para gerar as predições.")
        return
        
    resultados = []
    
    for caminho in arquivos_predicoes:
        nome_arquivo = os.path.basename(caminho)
        segmento = nome_arquivo.replace('predicoes_', '').replace('_wf.csv', '')
        
        df = pd.read_csv(caminho)
            
        pval_acc = calcular_mcnemar_pareado(df['Alvo'], df['BaselineTCN_Pred'], df['MSRTCN_Pred'])
        pval_f1, f1_base, f1_msr = calcular_bootstrap_f1(df['Alvo'], df['BaselineTCN_Pred'], df['MSRTCN_Pred'])
        acc_msrtcn = accuracy_score(df['Alvo'], df['MSRTCN_Pred'])
        acc_baseline = accuracy_score(df['Alvo'], df['BaselineTCN_Pred'])
        
        resultados.append({
            'Segmento': segmento,
            'Amostras (N)': len(df),
            'Acurácia TCN Controle': f"{acc_baseline*100:.2f}%",
            'Acurácia MSR-TCN': f"{acc_msrtcn*100:.2f}%",
            'F1 TCN Controle': f"{f1_base:.4f}",
            'F1 MSR-TCN': f"{f1_msr:.4f}",
            'p_value_Acc': pval_acc,
            'p_value_F1': pval_f1,
            'MSR_Venceu_F1_Com_Significancia': 'Sim' if pval_f1 < 0.05 else 'Não'
        })
        
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        caminho_csv = os.path.join(diretorio_analises, 'testes_hipotese_segmentos.csv')
        df_resultados.to_csv(caminho_csv, index=False)
        
        print("\n--- Relatório de Testes de Hipótese ---")
        print(df_resultados.to_string(index=False))
        
        plotar_p_valores(df_resultados, diretorio_analises)
        print(f"\nRelatório de testes estatísticos salvo em {caminho_csv}")

if __name__ == '__main__':
    main()
