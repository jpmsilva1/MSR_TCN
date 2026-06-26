"""
Script de Otimização de Hiperparâmetros para o MSR-TCN.
Executa uma busca em grade (grid search) para encontrar os hiperparâmetros e 
estratégias de Aumentação de Dados (Data Augmentation) ideais para cada um dos 8 segmentos de mercado.
"""
import os
import json
import itertools
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
# Garante que os módulos em src possam ser importados
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from configuracoes import SEGMENTOS, DIRETORIO_DADOS, DIRETORIO_CONFIGS
from modelos import MSRTCN1D
from utilidades import SegmentTimeSeriesDataset, FocalLoss

def otimizar_segmento(nome_segmento: str, ativos_segmento: list, device: torch.device):
    print(f"\n[{nome_segmento}] Carregando dados do segmento...")
    arquivos_completos = [os.path.join(DIRETORIO_DADOS, f"{t.replace('^', '').replace('=', '_')}_full.csv") for t in ativos_segmento]
    
    # Utilizamos um período restrito (2013-2014) para uma busca de hiperparâmetros rápida.
    # A avaliação real utiliza a validação Walk-Forward de 2015 a 2024.
    dataset_treino = SegmentTimeSeriesDataset(arquivos_completos, is_train=True, start_date='2013-01-01', end_date='2013-12-31')
    if len(dataset_treino) == 0:
        print(f"[{nome_segmento}] Sem dados de otimização disponíveis. Ignorando.")
        return None
        
    dataset_validacao = SegmentTimeSeriesDataset(arquivos_completos, is_train=False, start_date='2014-01-01', end_date='2014-12-31', scaler=dataset_treino.scaler)
    
    loader_treino = DataLoader(dataset_treino, batch_size=128, shuffle=True)
    loader_validacao = DataLoader(dataset_validacao, batch_size=128, shuffle=False)
        
    # Pré-calcula os pesos alpha para a Focal Loss com base na distribuição de classes
    alvos_treino = torch.tensor(dataset_treino.y, dtype=torch.long)
    contagens = torch.bincount(alvos_treino, minlength=3).float()
    contagens[contagens == 0] = 1.0
    pesos_alpha = len(dataset_treino.y) / (3.0 * contagens)
    pesos_alpha = pesos_alpha.to(device)
        
    grade = {
        'kernel_size': [3, 5],
        'dropout': [0.2],
        'learning_rate': [1e-3, 5e-4],
        'gamma': [2, 3],
        'use_jitter': [True, False],
        'use_warp': [True, False],
        'use_slice': [True, False],
        'use_mixup': [True, False],
        'use_decomp': [True, False]
    }
    
    chaves = grade.keys()
    combinacoes = list(itertools.product(*grade.values()))
    resultados = []
    
    print(f"[{nome_segmento}] Testando {len(combinacoes)} combinações na grade...")
    
    for idx, combo in enumerate(tqdm(combinacoes, desc=f"Grade {nome_segmento}", leave=False)):
        parametros = dict(zip(chaves, combo))
        dataset_treino.set_da_flags(parametros)
        
        modelo = MSRTCN1D(in_channels=6, seq_len=32, kernel_size=parametros['kernel_size'], dropout=parametros['dropout']).to(device)
        criterio = FocalLoss(alpha=pesos_alpha, gamma=parametros['gamma'])
        otimizador = optim.AdamW(modelo.parameters(), lr=parametros['learning_rate'], weight_decay=1e-4)
        
        melhor_perda_val = float('inf')
        
        # 5 épocas para avaliação rápida na busca em grade
        for epoca in range(5):
            modelo.train()
            for X, y in loader_treino:
                X, y = X.to(device), y.to(device)
                otimizador.zero_grad()
                saidas = modelo(X)
                perda = criterio(saidas, y)
                perda.backward()
                otimizador.step()
                
            modelo.eval()
            perda_val_epoca = 0.0
            with torch.no_grad():
                for X, y in loader_validacao:
                    X, y = X.to(device), y.to(device)
                    saidas = modelo(X)
                    perda = criterio(saidas, y)
                    perda_val_epoca += perda.item() * X.size(0)
            
            perda_val_epoca /= len(dataset_validacao)
            if perda_val_epoca < melhor_perda_val: 
                melhor_perda_val = perda_val_epoca
            
        resultados.append({
            **parametros,
            'val_loss': melhor_perda_val 
        })
        
    df_resultados = pd.DataFrame(resultados).sort_values(by='val_loss', ascending=True)
    
    # Salva a melhor configuração
    melhores_parametros = df_resultados.iloc[0].to_dict()
    caminho_json_melhor = os.path.join(DIRETORIO_CONFIGS, f'melhores_parametros_{nome_segmento}.json')
    
    parametros_finais = {
        **melhores_parametros,
        'seq_len': 32,
        'in_channels': 6
    }
    del parametros_finais['val_loss']
    
    with open(caminho_json_melhor, 'w') as f:
        json.dump(parametros_finais, f, indent=4)
        
    print(f"[{nome_segmento}] Melhor configuração salva em: {caminho_json_melhor}")
    return parametros_finais

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Iniciando Otimização de Hiperparâmetros usando o dispositivo: {device}")
    
    todos_melhores_parametros = []
    
    for nome_segmento, ativos in SEGMENTOS.items():
        res = otimizar_segmento(nome_segmento, ativos, device)
        if res is not None:
            res['Segmento'] = nome_segmento
            todos_melhores_parametros.append(res)
            
    if todos_melhores_parametros:
        df_resumo = pd.DataFrame(todos_melhores_parametros)
        colunas = ['Segmento'] + [c for c in df_resumo.columns if c != 'Segmento']
        df_resumo = df_resumo[colunas]
        
        # Salva o resumo
        diretorio_resultados = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(diretorio_resultados, exist_ok=True)
        caminho_csv = os.path.join(diretorio_resultados, 'resumo_otimizacao.csv')
        df_resumo.to_csv(caminho_csv, index=False)
        print(f"\n[Sucesso] Resumo da otimização salvo em: {caminho_csv}")

if __name__ == "__main__":
    main()
