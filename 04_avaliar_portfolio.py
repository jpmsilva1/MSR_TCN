"""
Script de Avaliação Financeira de Portfólio.
Calcula métricas financeiras institucionais (ROI, Log-Retorno, Sharpe, Sortino, Calmar e MDD)
para simular a aplicação das predições do MSR-TCN na gestão de um fundo de investimento.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from configuracoes import DIRETORIO_RESULTADOS

DIRETORIO_PORTFOLIO = os.path.join(DIRETORIO_RESULTADOS, 'avaliacao_portfolio')

def configurar_estilo():
    """Configurações visuais institucionais para os gráficos de desempenho."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20

def calcular_sharpe(curva_patrimonio, dias_trade=252):
    """Calcula o Índice de Sharpe anualizado assumindo Taxa Livre de Risco = 0"""
    retornos_diarios = np.diff(curva_patrimonio) / curva_patrimonio[:-1]
    desvio = np.std(retornos_diarios)
    if desvio == 0:
        return 0
    return (np.mean(retornos_diarios) / desvio) * np.sqrt(dias_trade)

def calcular_mdd(curva_patrimonio):
    """Calcula o Maximum Drawdown (MDD) absoluto em %."""
    maximo_acumulado = np.maximum.accumulate(curva_patrimonio)
    maximo_acumulado[maximo_acumulado == 0] = 1 # Proteção contra div/0
    quedas = (curva_patrimonio - maximo_acumulado) / maximo_acumulado
    return np.abs(np.min(quedas)) * 100

def calcular_sortino(curva_patrimonio, dias_trade=252):
    """Calcula o Índice de Sortino."""
    retornos_diarios = np.diff(curva_patrimonio) / curva_patrimonio[:-1]
    retornos_negativos = retornos_diarios[retornos_diarios < 0]
    
    desvio_negativo = np.std(retornos_negativos) if len(retornos_negativos) > 0 else 0
    if desvio_negativo == 0:
        return 0
        
    return (np.mean(retornos_diarios) / desvio_negativo) * np.sqrt(dias_trade)

def calcular_calmar(curva_patrimonio, dias_trade=252):
    """Calcula o Índice de Calmar: CAGR / MDD."""
    anos = len(curva_patrimonio) / dias_trade
    if anos == 0: return 0
    
    cagr = ((curva_patrimonio[-1] / curva_patrimonio[0]) ** (1/anos)) - 1
    mdd = calcular_mdd(curva_patrimonio) / 100.0  
    
    if mdd == 0:
        return 0
    return cagr / mdd

def simular_retornos(predicoes, retornos_1d, capital_inicial=10000.0, custo_transacao=0.0003):
    """
    Simula a curva de capital considerando custos de transação (corretagem/spread).
    Posição: 2=Comprado(Long), 1=Vendido(Flat/Hold em long-only).
    """
    n = len(predicoes)
    retornos_diarios_estrategia = np.zeros(n)
    curva_capital = np.zeros(n + 1)
    curva_capital[0] = capital_inicial
    
    posicao = 0.0  
    
    predicoes_arr = np.array(predicoes)
    ret_arr = np.array(retornos_1d) / 100.0
    
    for i in range(n):
        pred = predicoes_arr[i]
        posicao_anterior = posicao
        
        # Estratégia Long-Only: 2 = Comprar, 1 = Vender (Ficar de fora)
        if pred == 2: posicao = 1.0
        elif pred == 1: posicao = 0.0
        
        custo = custo_transacao if posicao != posicao_anterior else 0.0
        retorno_diario = (posicao * ret_arr[i]) - custo
        retornos_diarios_estrategia[i] = retorno_diario
        curva_capital[i+1] = curva_capital[i] * (1 + retorno_diario)
        
    return retornos_diarios_estrategia, curva_capital[1:]

def plotar_crescimento_portfolio(datas, curva_msr, curva_base, curva_bh, nome_segmento, sufixo_titulo=""):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(datas, curva_bh, label='Mercado (Buy & Hold)', color='#B0BEC5', linestyle='--', linewidth=2)
    ax.plot(datas, curva_base, label='Estratégia TCN Controle', color='#F24236', linewidth=2, alpha=0.8)
    ax.plot(datas, curva_msr, label='Estratégia MSR-TCN', color='#2E86AB', linewidth=3)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'R$ {x:,.0f}'.replace(',', '.')))
    ax.set_title(f'Crescimento de Fundo de Investimento: Portfólio {nome_segmento}\n(Capital Inicial R$ 10.000 por Ativo, Ponderação Igualitária){sufixo_titulo}', fontweight='bold', pad=20)
    ax.set_ylabel('Patrimônio Total do Portfólio', fontweight='bold')
    ax.set_xlabel('Período de Validação', fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIRETORIO_PORTFOLIO, f'crescimento_portfolio_{nome_segmento}.png'), dpi=300)
    plt.close()

def plotar_barras_comparacao(df_metricas, sufixo_metrica, string_titulo, string_ylabel, nome_saida, is_percentagem=True):
    plt.figure(figsize=(12, 8))
    
    colunas_plotar = [f'Buy&Hold_{sufixo_metrica}', f'Baseline_{sufixo_metrica}', f'MSRTCN_{sufixo_metrica}']
    
    plot_df = pd.melt(df_metricas, id_vars=['Segmento'], value_vars=colunas_plotar,
                      var_name='Estratégia', value_name='Valor')
    
    plot_df['Estratégia'] = plot_df['Estratégia'].map({
        f'Buy&Hold_{sufixo_metrica}': 'Buy & Hold',
        f'Baseline_{sufixo_metrica}': 'TCN Controle',
        f'MSRTCN_{sufixo_metrica}': 'MSR-TCN'
    })
    
    ax = sns.barplot(x='Segmento', y='Valor', hue='Estratégia', data=plot_df, palette=['#B0BEC5', '#F24236', '#2E86AB'])
    
    plt.title(string_titulo, fontweight='bold', pad=20)
    plt.ylabel(string_ylabel, fontweight='bold')
    plt.xlabel('Segmento', fontweight='bold')
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height != 0:
            fmt = f'{height:.0f}%' if is_percentagem else f'{height:.2f}'
            ax.annotate(fmt, (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 5),
                        textcoords='offset points')
                        
    plt.legend(title='Estratégia', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRETORIO_PORTFOLIO, nome_saida), dpi=300)
    plt.close()

def main():
    configurar_estilo()
    os.makedirs(DIRETORIO_PORTFOLIO, exist_ok=True)
    print("Gerando Avaliação Financeira Institucional (Log-Retorno, Sharpe, MDD, Sortino, Calmar)...")
    
    segmentos_vencedores = ['SmallCaps', 'FIIs', 'MegaCapsTech', 'TradicionaisGlobais', 'CambioGlobal']
    capital_inicial_por_ativo = 10000.0
    
    todas_metricas = []
    
    for segmento in segmentos_vencedores:
        caminho_csv = os.path.join(DIRETORIO_RESULTADOS, f"predicoes_{segmento}_wf.csv")
        if not os.path.exists(caminho_csv):
            print(f"  [{segmento}] Arquivo {caminho_csv} não encontrado. Execute o script de treino.")
            continue
            
        print(f"  Analisando portfólio {segmento}...")
        df = pd.read_csv(caminho_csv)
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values(['Ativo', 'Data']).reset_index(drop=True)
        
        ativos = df['Ativo'].unique()
        datas_portfolio = np.sort(df['Data'].unique())
        
        portfolio_msr = np.zeros(len(datas_portfolio))
        portfolio_base = np.zeros(len(datas_portfolio))
        portfolio_bh = np.zeros(len(datas_portfolio))
        
        portfolio_msr_sem_custo = np.zeros(len(datas_portfolio))
        portfolio_base_sem_custo = np.zeros(len(datas_portfolio))
        
        metricas_ativos = []
        
        for ativo in ativos:
            df_ativo = df[df['Ativo'] == ativo].copy()
            df_ativo = df_ativo.set_index('Data').reindex(datas_portfolio).fillna({'Ret_1d': 0, 'MSRTCN_Pred': 0, 'BaselineTCN_Pred': 0})
            
            retorno_bh = df_ativo['Ret_1d'].values / 100.0
            curva_bh = capital_inicial_por_ativo * np.cumprod(1 + retorno_bh)
            
            _, curva_msr = simular_retornos(df_ativo['MSRTCN_Pred'].values, df_ativo['Ret_1d'].values, capital_inicial_por_ativo, 0.0003)
            _, curva_base = simular_retornos(df_ativo['BaselineTCN_Pred'].values, df_ativo['Ret_1d'].values, capital_inicial_por_ativo, 0.0003)
            
            _, curva_msr_sem_custo = simular_retornos(df_ativo['MSRTCN_Pred'].values, df_ativo['Ret_1d'].values, capital_inicial_por_ativo, 0.0)
            _, curva_base_sem_custo = simular_retornos(df_ativo['BaselineTCN_Pred'].values, df_ativo['Ret_1d'].values, capital_inicial_por_ativo, 0.0)
            
            portfolio_msr += curva_msr
            portfolio_base += curva_base
            portfolio_bh += curva_bh
            portfolio_msr_sem_custo += curva_msr_sem_custo
            portfolio_base_sem_custo += curva_base_sem_custo
            
            metricas_ativos.append({
                'Segmento': segmento,
                'Ativo': ativo,
                'Buy&Hold_ROI(%)': ((curva_bh[-1] - capital_inicial_por_ativo) / capital_inicial_por_ativo) * 100,
                'Baseline_ROI(%)': ((curva_base[-1] - capital_inicial_por_ativo) / capital_inicial_por_ativo) * 100,
                'MSRTCN_ROI(%)': ((curva_msr[-1] - capital_inicial_por_ativo) / capital_inicial_por_ativo) * 100,
                'Buy&Hold_Log': np.log(curva_bh[-1] / capital_inicial_por_ativo),
                'Baseline_Log': np.log(max(curva_base[-1], 1e-5) / capital_inicial_por_ativo),
                'MSRTCN_Log': np.log(max(curva_msr[-1], 1e-5) / capital_inicial_por_ativo)
            })
            
        plotar_crescimento_portfolio(datas_portfolio, portfolio_msr, portfolio_base, portfolio_bh, segmento)
        plotar_crescimento_portfolio(datas_portfolio, portfolio_msr_sem_custo, portfolio_base_sem_custo, portfolio_bh, f"{segmento}_sem_custos", "\n[SEM TAXA DE CORRETAGEM]")
        
        df_ativos = pd.DataFrame(metricas_ativos)
        
        todas_metricas.append({
            'Segmento': segmento,
            'Ativos': len(ativos),
            'Buy&Hold_ROI_Mean(%)': df_ativos['Buy&Hold_ROI(%)'].mean(),
            'Baseline_ROI_Mean(%)': df_ativos['Baseline_ROI(%)'].mean(),
            'MSRTCN_ROI_Mean(%)': df_ativos['MSRTCN_ROI(%)'].mean(),
            
            'Buy&Hold_Log_Mean': df_ativos['Buy&Hold_Log'].mean(),
            'Baseline_Log_Mean': df_ativos['Baseline_Log'].mean(),
            'MSRTCN_Log_Mean': df_ativos['MSRTCN_Log'].mean(),
            
            'Buy&Hold_Sharpe': calcular_sharpe(portfolio_bh),
            'Baseline_Sharpe': calcular_sharpe(portfolio_base),
            'MSRTCN_Sharpe': calcular_sharpe(portfolio_msr),
            
            'Buy&Hold_Sortino': calcular_sortino(portfolio_bh),
            'Baseline_Sortino': calcular_sortino(portfolio_base),
            'MSRTCN_Sortino': calcular_sortino(portfolio_msr),
            
            'Buy&Hold_MDD(%)': calcular_mdd(portfolio_bh),
            'Baseline_MDD(%)': calcular_mdd(portfolio_base),
            'MSRTCN_MDD(%)': calcular_mdd(portfolio_msr),
            
            'Buy&Hold_Calmar': calcular_calmar(portfolio_bh),
            'Baseline_Calmar': calcular_calmar(portfolio_base),
            'MSRTCN_Calmar': calcular_calmar(portfolio_msr)
        })
        
        df_ativos.to_csv(os.path.join(DIRETORIO_PORTFOLIO, f'roi_detalhado_{segmento}.csv'), index=False)
        
    if not todas_metricas:
        print("Nenhuma métrica computada. Verifique os arquivos CSV de predição.")
        return

    df_metricas = pd.DataFrame(todas_metricas)
    df_metricas.to_csv(os.path.join(DIRETORIO_PORTFOLIO, 'resumo_segmentos_roi.csv'), index=False)
    
    # Gerar os gráficos de barra
    plotar_barras_comparacao(df_metricas, 'ROI_Mean(%)', 'Retorno Total (Média Aritmética) por Segmento', 'ROI Médio (%)', 'comparacao_segmento_roi_media.png', True)
    plotar_barras_comparacao(df_metricas, 'Log_Mean', 'Log-Retorno Médio (Penalização de Outliers)', 'Média do Log-Retorno', 'comparacao_segmento_roi_log.png', False)
    plotar_barras_comparacao(df_metricas, 'Sharpe', 'Sharpe Ratio Institucional (Retorno/Volatilidade Total)', 'Sharpe Ratio', 'comparacao_segmento_sharpe.png', False)
    plotar_barras_comparacao(df_metricas, 'Sortino', 'Sortino Ratio (Retorno / Volatilidade Negativa)', 'Sortino Ratio', 'comparacao_segmento_sortino.png', False)
    plotar_barras_comparacao(df_metricas, 'Calmar', 'Calmar Ratio (Retorno Anual / MDD)', 'Calmar Ratio', 'comparacao_segmento_calmar.png', False)
    # MDD plot deve ter formatação de porcentagem
    plotar_barras_comparacao(df_metricas, 'MDD(%)', 'Maximum Drawdown (Risco de Ruína) - Menor é Melhor', 'MDD (%)', 'comparacao_segmento_mdd.png', True)
    
    print(f"\n✅ Relatórios Institucionais de Portfólio concluídos: {DIRETORIO_PORTFOLIO}")

if __name__ == '__main__':
    main()
