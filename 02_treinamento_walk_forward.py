"""
Script de Treinamento de Validação Cruzada Walk-Forward para o MSR-TCN.
Avalia o modelo ao longo de um período de 10 anos (2015-2024) utilizando uma abordagem
de janela em expansão (expanding window) para evitar vazamento de dados.
"""
import os
import json
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from configuracoes import SEGMENTOS, DIRETORIO_DADOS, DIRETORIO_CONFIGS, DIRETORIO_CHECKPOINTS, DIRETORIO_RESULTADOS
from modelos import MSRTCN1D, BaselineTCN1D
from utilidades import FocalLoss, SegmentTimeSeriesDataset


def treinar_e_avaliar_modelo(modelo, loader_treino, loader_validacao, loader_teste, device, criterio, 
                             epocas=10, caminho_salvar='modelo.pt', lr=1e-3, caminho_plotar=None):
    """
    Loop de treinamento padrão com parada antecipada na validação (salvando o melhor modelo).
    """
    otimizador = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    melhor_perda_val = float('inf')
    
    perdas_treino = []
    perdas_validacao = []
    
    pbar = tqdm(range(epocas), desc=f"Treinando {os.path.basename(caminho_salvar)}", leave=False)
    for epoca in pbar:
        modelo.train()
        perda_treino_epoca = 0.0
        for X, y in loader_treino:
            X, y = X.to(device), y.to(device)
            otimizador.zero_grad()
            saidas = modelo(X)
            perda = criterio(saidas, y)
            perda.backward()
            otimizador.step()
            perda_treino_epoca += perda.item() * X.size(0)
            
        perda_treino_epoca /= len(loader_treino.dataset)
        perdas_treino.append(perda_treino_epoca)
            
        modelo.eval()
        perda_val_epoca = 0.0
        with torch.no_grad():
            for X, y in loader_validacao:
                X, y = X.to(device), y.to(device)
                saidas = modelo(X)
                perda = criterio(saidas, y)
                perda_val_epoca += perda.item() * X.size(0)
        
        perda_val_epoca /= len(loader_validacao.dataset)
        perdas_validacao.append(perda_val_epoca)
        
        pbar.set_postfix({'T_Loss': f"{perda_treino_epoca:.4f}", 'V_Loss': f"{perda_val_epoca:.4f}"})
        
        if perda_val_epoca < melhor_perda_val:
            melhor_perda_val = perda_val_epoca
            torch.save(modelo.state_dict(), caminho_salvar)
            
    if caminho_plotar:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(caminho_plotar), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epocas + 1), perdas_treino, label='Treino', color='#2E86AB', linewidth=2)
        plt.plot(range(1, epocas + 1), perdas_validacao, label='Validação', color='#F24236', linewidth=2, linestyle='--')
        plt.title('Curvas de Aprendizado (Loss vs Épocas)', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Focal Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(caminho_plotar, dpi=300)
        plt.close()
            
    # Avaliação final no conjunto de Teste
    modelo.load_state_dict(torch.load(caminho_salvar))
    modelo.eval()
    todas_preds, todos_alvos = [], []
    with torch.no_grad():
        for X, y in loader_teste:
            X, y = X.to(device), y.to(device)
            saidas = modelo(X)
            _, previso = torch.max(saidas, 1)
            todas_preds.extend(previso.cpu().numpy())
            todos_alvos.extend(y.cpu().numpy())
            
    acc = accuracy_score(todos_alvos, todas_preds)
    f1 = f1_score(todos_alvos, todas_preds, average='macro')
    return acc, f1, todas_preds, todos_alvos


def treinar_e_avaliar_tcn(nome_modelo, segmento, loader_treino, loader_validacao, loader_teste, device, pesos_alpha=None, melhores_parametros=None):
    print(f"  Treinando {nome_modelo} para {segmento}...")
    
    # Fallback para os parâmetros padrão se as configurações estiverem faltando
    if melhores_parametros is None:
        melhores_parametros = {'kernel_size': 3, 'dropout': 0.2, 'learning_rate': 1e-3}
        
    if nome_modelo == 'MSRTCN':
        modelo = MSRTCN1D(
            in_channels=6, 
            seq_len=32,
            kernel_size=melhores_parametros.get('kernel_size', 3),
            dropout=melhores_parametros.get('dropout', 0.2)
        ).to(device)
    else:
        # TCN Baseline com quantidade equivalente de parâmetros
        modelo = BaselineTCN1D(
            in_channels=6,
            seq_len=32,
            kernel_size=melhores_parametros.get('kernel_size', 3),
            dropout=melhores_parametros.get('dropout', 0.2)
        ).to(device)
    
    criterio = FocalLoss(gamma=2, alpha=pesos_alpha)
    
    caminho_salvar = os.path.join(DIRETORIO_CHECKPOINTS, f"{nome_modelo}_{segmento}.pt")
    caminho_plotar = os.path.join(DIRETORIO_RESULTADOS, 'curvas_aprendizado', f"ca_{nome_modelo}_{segmento}.png")
    
    return treinar_e_avaliar_modelo(
        modelo, loader_treino, loader_validacao, loader_teste, device, criterio, 
        epocas=10, caminho_salvar=caminho_salvar, lr=melhores_parametros.get('learning_rate', 1e-3),
        caminho_plotar=caminho_plotar
    )

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Iniciando Experimento MSR-TCN Walk-Forward (Dispositivo: {device})")
    
    resultados = []
    
    # No artigo original, 5 dos 8 segmentos mostraram um desempenho significativamente superior
    segmentos_vencedores = ['SmallCaps', 'FIIs', 'MegaCapsTech', 'TradicionaisGlobais', 'CambioGlobal']
    segmentos_alvo = {k: v for k, v in SEGMENTOS.items() if k in segmentos_vencedores}
    
    for segmento, ativos in tqdm(segmentos_alvo.items(), desc="Segmentos"):
        print(f"\n--- Segmento: {segmento} ---")
        arquivos_completos = [os.path.join(DIRETORIO_DADOS, f"{t.replace('^', '').replace('=', '_')}_full.csv") for t in ativos]
        
        caminho_json = os.path.join(DIRETORIO_CONFIGS, f"melhores_parametros_{segmento}.json")
        melhores_parametros = None
        if os.path.exists(caminho_json):
            with open(caminho_json, 'r') as f:
                melhores_parametros = json.load(f)
            print(f"  Parâmetros otimizados carregados: LR={melhores_parametros.get('learning_rate')}, Kernel={melhores_parametros.get('kernel_size')}")
        else:
            print(f"  Aviso: {caminho_json} não encontrado. Usando parâmetros padrão.")
            
        todos_alvos_msrtcn, todas_preds_msrtcn = [], []
        todos_alvos_baseline, todas_preds_baseline = [], []
        
        todas_datas, todos_ativos, todos_ret_1d = [], [], []
        
        # Validação Walk-Forward: 2015 a 2024
        # Treino: [ano_teste - 6] a [ano_teste - 2]
        # Validação: [ano_teste - 1]
        # Teste: [ano_teste]
        for ano_teste in range(2015, 2025):
            print(f"  [Walk-Forward] Ano de Teste: {ano_teste}")
            
            inicio_treino = f"{ano_teste - 6}-01-01"
            fim_treino = f"{ano_teste - 2}-12-31"
            inicio_val = f"{ano_teste - 1}-01-01"
            fim_val = f"{ano_teste - 1}-12-31"
            inicio_teste = f"{ano_teste}-01-01"
            fim_teste = f"{ano_teste}-12-31"
            
            dataset_treino = SegmentTimeSeriesDataset(arquivos_completos, is_train=True, start_date=inicio_treino, end_date=fim_treino)
            if len(dataset_treino) == 0: continue
                
            dataset_val = SegmentTimeSeriesDataset(arquivos_completos, is_train=False, start_date=inicio_val, end_date=fim_val, scaler=dataset_treino.scaler)
            dataset_teste = SegmentTimeSeriesDataset(arquivos_completos, is_train=False, start_date=inicio_teste, end_date=fim_teste, scaler=dataset_treino.scaler)
            
            if len(dataset_val) == 0 or len(dataset_teste) == 0: continue
                
            if melhores_parametros: 
                dataset_treino.set_da_flags(melhores_parametros)
                
            todas_datas.extend(dataset_teste.dates)
            todos_ativos.extend(dataset_teste.tickers)
            todos_ret_1d.extend(dataset_teste.ret_1d)
                
            loader_treino = DataLoader(dataset_treino, batch_size=128, shuffle=True)
            loader_val = DataLoader(dataset_val, batch_size=128, shuffle=False)
            loader_teste = DataLoader(dataset_teste, batch_size=128, shuffle=False)
            
            # Pesos dinâmicos da Focal Loss
            y_tensor = torch.tensor(dataset_treino.y, dtype=torch.long)
            contagens = torch.bincount(y_tensor, minlength=3).float()
            contagens[contagens == 0] = 1.0
            pesos_alpha = len(dataset_treino.y) / (3.0 * contagens)
            pesos_alpha = pesos_alpha.to(device)
            
            acc_m, f1_m, p_m, t_m = treinar_e_avaliar_tcn('MSRTCN', f"{segmento}_{ano_teste}", loader_treino, loader_val, loader_teste, device, pesos_alpha, melhores_parametros)
            acc_b, f1_b, p_b, t_b = treinar_e_avaliar_tcn('BaselineTCN', f"{segmento}_{ano_teste}", loader_treino, loader_val, loader_teste, device, pesos_alpha, melhores_parametros)
            
            todas_preds_msrtcn.extend(p_m)
            todos_alvos_msrtcn.extend(t_m)
            todas_preds_baseline.extend(p_b)
            todos_alvos_baseline.extend(t_b)
            
        if len(todos_alvos_msrtcn) == 0: continue
            
        acc_msrtcn = accuracy_score(todos_alvos_msrtcn, todas_preds_msrtcn)
        f1_msrtcn = f1_score(todos_alvos_msrtcn, todas_preds_msrtcn, average='macro')
        acc_baseline = accuracy_score(todos_alvos_baseline, todas_preds_baseline)
        f1_baseline = f1_score(todos_alvos_baseline, todas_preds_baseline, average='macro')
        
        # Salva as predições para avaliação estatística e análise de portfólio
        df_segmento = pd.DataFrame({
            'Data': todas_datas,
            'Ativo': todos_ativos,
            'Ret_1d': todos_ret_1d,
            'Alvo': todos_alvos_msrtcn,
            'BaselineTCN_Pred': todas_preds_baseline,
            'MSRTCN_Pred': todas_preds_msrtcn
        })
        df_segmento.to_csv(os.path.join(DIRETORIO_RESULTADOS, f'predicoes_{segmento}_wf.csv'), index=False)
        
        resultados.append({
            'Segmento': segmento,
            'BaselineTCN_Acc': acc_baseline, 'BaselineTCN_F1': f1_baseline,
            'MSRTCN_Acc': acc_msrtcn, 'MSRTCN_F1': f1_msrtcn
        })
        
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(os.path.join(DIRETORIO_RESULTADOS, 'resumo_metricas_walk_forward.csv'), index=False)
    
    print("\nResultados Globais MSR-TCN (Walk-Forward):")
    print(df_resultados)
    print("\nTreinamento Walk-Forward concluído com sucesso!")

if __name__ == '__main__':
    main()
