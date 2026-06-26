"""
Script de Benchmark de Custo Computacional.
Compara empiricamente e teoricamente o custo computacional (MACs, Parâmetros e Latência)
entre a arquitetura proposta MSR-TCN e o baseline TCN de Controle.
"""
import os
import torch
import time
import thop

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from configuracoes import DIRETORIO_RESULTADOS
from modelos import MSRTCN1D, BaselineTCN1D

def medir_macs_e_parametros(modelo, formato_entrada):
    """Mede Operações de Multiplicação e Acumulação (MACs) e número de parâmetros."""
    modelo.eval()
    modelo.cpu()
    entrada_falsa = torch.randn(formato_entrada)
    macs, parametros = thop.profile(modelo, inputs=(entrada_falsa,), verbose=False)
    return macs, parametros

def medir_tempo_inferencia(modelo_fn, device, formato_entrada, num_execucoes=10000):
    """Mede a latência empírica e a vazão (throughput) de inferência no hardware atual."""
    modelo = modelo_fn().to(device)
    modelo.eval()
    entrada_falsa = torch.randn(formato_entrada).to(device)
    
    # Aquecimento (Warm-up)
    with torch.no_grad():
        for _ in range(100):
            _ = modelo(entrada_falsa)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
        
    tempo_inicio = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_execucoes):
            _ = modelo(entrada_falsa)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
        
    tempo_fim = time.perf_counter()
    
    tempo_medio_ms = ((tempo_fim - tempo_inicio) / num_execucoes) * 1000
    tamanho_lote = formato_entrada[0]
    vazao = tamanho_lote / (tempo_medio_ms / 1000)
    return tempo_medio_ms, vazao

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Iniciando Benchmark de Eficiência Computacional (Dispositivo: {device})")
    formato_entrada = (128, 6, 32)
    
    # 1. Complexidade Teórica
    macs_base, params_base = medir_macs_e_parametros(BaselineTCN1D(in_channels=6, seq_len=32), formato_entrada)
    macs_msr, params_msr = medir_macs_e_parametros(MSRTCN1D(in_channels=6, seq_len=32), formato_entrada)
    
    # 2. Latência Empírica
    tempo_base, vazao_base = medir_tempo_inferencia(lambda: BaselineTCN1D(in_channels=6, seq_len=32), device, formato_entrada, num_execucoes=5000)
    tempo_msr, vazao_msr = medir_tempo_inferencia(lambda: MSRTCN1D(in_channels=6, seq_len=32), device, formato_entrada, num_execucoes=5000)
    
    relatorio = f"""====================================================
RELATÓRIO DE BENCHMARK COMPUTACIONAL
====================================================

1. Complexidade Teórica (Independente de Hardware)
----------------------------------------------------
[TCN Controle] Parâmetros: {params_base:,.0f} | MACs: {macs_base:,.0f}
[MSR-TCN]      Parâmetros: {params_msr:,.0f} | MACs: {macs_msr:,.0f}

> Redução de Parâmetros: {(1 - params_msr/params_base)*100:.2f}%
> Redução de MACs:       {(1 - macs_msr/macs_base)*100:.2f}%

2. Latência Empírica (Teste de Stress, Lote=128)
----------------------------------------------------
Dispositivo de Execução: {device.type.upper()}

[TCN Controle] Tempo de Inferência: {tempo_base:.3f} ms | Vazão: {vazao_base:,.0f} sequências/seg
[MSR-TCN]      Tempo de Inferência: {tempo_msr:.3f} ms | Vazão: {vazao_msr:,.0f} sequências/seg

> Melhoria de Velocidade: {(tempo_base/tempo_msr - 1)*100:.2f}% mais rápido
====================================================
"""
    print(relatorio)
    
    os.makedirs(DIRETORIO_RESULTADOS, exist_ok=True)
    caminho_salvar = os.path.join(DIRETORIO_RESULTADOS, 'relatorio_benchmark_computacional.txt')
    with open(caminho_salvar, 'w') as f:
        f.write(relatorio)
        
    print(f"✅ Relatório de Benchmark salvo em: {caminho_salvar}")

if __name__ == "__main__":
    main()
