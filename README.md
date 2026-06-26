# MSR-TCN: Rede Convolucional Temporal Residual Multiescala com Decomposição Adaptativa para Previsão Financeira

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositório contém o código-fonte oficial e o pipeline de reprodutibilidade para o modelo **MSR-TCN (Multi-Scale Residual Temporal Convolutional Network)**, uma arquitetura de *Deep Learning* proposta para solucionar problemas de ruído de alta frequência e não-estacionariedade inerentes à previsão de séries temporais financeiras.

## 🎯 Resumo

O mercado financeiro gera dados com um altíssimo grau de estocasticidade (ruído) e não-estacionariedade (mudança de distribuições ao longo do tempo). Redes Neurais convencionais (como CNNs e LSTMs) frequentemente sofrem de *overfitting* ou falham em isolar a verdadeira tendência de longo prazo das flutuações voláteis intradiárias.

A **MSR-TCN** resolve este problema ao integrar uma **Decomposição Adaptativa em Subbandas (ASD) 1D Assimétrica** nativa à arquitetura, fragmentando o sinal financeiro bruto em três frequências distintas:
1. **Alta Frequência (Ruído):** Captura a volatilidade imediata.
2. **Média Frequência (Sazonalidade):** Captura ciclos intermédios de mercado.
3. **Baixa Frequência (Tendência):** Isola a direção subjacente do ativo.

Cada subbanda é processada por blocos isolados de **Redes Convolucionais Temporais (TCN)** com convoluções causais e dilatadas, permitindo um amplo campo receptivo (*receptive field*) sem perda de granularidade temporal.

---

## 🧠 Arquitetura do Modelo

A arquitetura ponta a ponta (*end-to-end*) do modelo processa vetores de características (Preço, Volume, Indicadores Técnicos) diretamente na série temporal.

```mermaid
graph TD
    classDef input fill:#2E86AB,stroke:#333,stroke-width:2px,color:#fff;
    classDef asd fill:#F24236,stroke:#333,stroke-width:2px,color:#fff;
    classDef tcn fill:#388E3C,stroke:#333,stroke-width:2px,color:#fff;
    classDef output fill:#FF9800,stroke:#333,stroke-width:2px,color:#fff;

    X["Entrada Temporal\n[Batch, Features=6, Seq=32]"]:::input

    subgraph "Asymmetric 1D Adaptive Subband Decomposition (ASD)"
        L1_U["Filtro L1 (Alta Freq.)\nMax Pooling"]:::asd
        L1_L["Filtro L1 (Baixa Freq.)\nMax Pooling"]:::asd
        
        L2_U["Filtro L2 (Média Freq.)\nMax Pooling"]:::asd
        L2_L["Filtro L2 (Baixa Freq.)\nMax Pooling"]:::asd
        
        Sync["Max Pooling\n(Sincronização Temporal)"]:::asd
    end

    subgraph "Temporal Convolutional Networks (TCNs Independentes)"
        TCN_Noise["TCN Subbanda Ruído\n(Dilatações: 1, 2)"]:::tcn
        TCN_Season["TCN Subbanda Sazonal\n(Dilatações: 1, 2)"]:::tcn
        TCN_Trend["TCN Subbanda Tendência\n(Dilatações: 1, 2)"]:::tcn
    end

    Concat["Concatenação de Features\n(Dim=96)"]:::input
    FC["Fully Connected (MLP)\n+ Dropout + ReLU"]:::output
    Out["Previsão (Softmax)\n[COMPRAR, VENDER, MANTER]"]:::output

    X --> L1_U
    X --> L1_L
    L1_L --> L2_U
    L1_L --> L2_L
    L1_U --> Sync

    Sync -->|Sinal Ruidoso| TCN_Noise
    L2_U -->|Sinal Sazonal| TCN_Season
    L2_L -->|Tendência Limpa| TCN_Trend

    TCN_Noise --> Concat
    TCN_Season --> Concat
    TCN_Trend --> Concat

    Concat --> FC --> Out
```

---

## 🔬 Metodologia de Validação (Walk-Forward)

Para assegurar rigor científico e total isolamento contra vazamento de dados (*data leakage* / *look-ahead bias*), este repositório adota a Validação Cruzada *Walk-Forward* com janelas de expansão.

```mermaid
gantt
    title Estratégia de Validação Walk-Forward em Expansão (2015-2024)
    dateFormat YYYY
    axisFormat %Y
    
    section Fold 1
    Treino (5 anos)      :active, t1, 2009, 2013
    Validação (1 ano)    :crit, v1, 2014, 2014
    Teste OOS (1 ano)    :done, f1, 2015, 2015

    section Fold 2
    Treino (6 anos)      :active, t2, 2009, 2014
    Validação (1 ano)    :crit, v2, 2015, 2015
    Teste OOS (1 ano)    :done, f2, 2016, 2016
    
    section Fold ...
    Treino               :active, t3, 2009, 2019
    Validação            :crit, v3, 2020, 2020
    Teste OOS            :done, f3, 2021, 2021
```

O conjunto de **Teste (Out-of-Sample)** avança 1 ano de cada vez. O modelo é inteiramente descartado e treinado do absoluto zero a cada iteração de teste, simulando com exatidão o ambiente produtivo e as mudanças de mercado (regimes de volatilidade).

---

## 🛠️ Pipeline de Execução MLOps

O repositório foi arquitetado sob princípios de modularidade, contendo uma trilha reprodutível e determinística ponta-a-ponta (do download do dado à avaliação financeira).

```mermaid
flowchart LR
    A[(API\nyfinance)] -->|coletar_dados.py| B[Data\nLake CSVs]
    B -->|01_otimizar.py| C{Grid Search\nHiperparâmetros}
    C -->|configs JSON| D(02_treinamento.py\nWalk-Forward CV)
    B --> D
    D -->|predicoes_wf.csv| E{Módulo de Avaliação}
    
    E -->|03_avaliar.py| F([Testes Estatísticos\nMcNemar / Bootstrap])
    E -->|04_portfolio.py| G([Backtest Financeiro\nSharpe, MDD, ROI])
    E -->|05_visualizacoes.py| H([Matrizes e Gráficos\nData Storytelling])
    D -->|06_benchmark.py| I([Benchmark Custo\nMACs / Latência])
```

### Passo a Passo da Replicação

1. **Instalação do Ambiente:**
   ```bash
   git clone https://github.com/SeuUsuario/MSR_TCN.git
   cd MSR_TCN
   pip install -r requirements.txt
   ```

2. **Ingestão de Dados e Feature Engineering:**
   ```bash
   python src/pipeline_dados/coletar_dados.py
   ```
   > Calcula características base (RSI, MACD, ATR) e rotula topos e fundos (BUY/SELL) usando janela centralizada.

3. **Otimização de Hiperparâmetros (Grid Search):**
   ```bash
   python 01_otimizar_hiperparametros.py
   ```
   > Determina hiperparâmetros ótimos (Kernel Size, Taxa de Aprendizado) e atalhos de *Data Augmentation* por segmento econômico, salvando em `configs/`.

4. **Treinamento e Inferência Walk-Forward:**
   ```bash
   python 02_treinamento_walk_forward.py
   ```
   > Executa a bateria exaustiva (10 anos) de testes fora da amostra (*Out-of-Sample*), gerando as predições.

5. **Testes Estatísticos e Backtest:**
   ```bash
   python 03_avaliar_estatisticas.py
   python 04_avaliar_portfolio.py
   ```
   > Analisa p-valores e métricas de fundos institucionais (Drawdown, Sortino, Calmar).

6. **Benchmark de Eficiência Computacional:**
   ```bash
   python 06_benchmark_custo_computacional.py
   ```
   > Comprova a eficiência (Número de Parâmetros, MACs e Throughput/Vazão) no processador atual (CPU / MPS / CUDA).

---

## 🧬 Técnicas de Aumentação de Dados (Data Augmentation)

O mercado financeiro apresenta um profundo desbalanceamento, onde os rótulos de `MANTER (HOLD)` compõem cerca de 90% da distribuição (vs. `COMPRAR` e `VENDER`).
As seguintes técnicas são dinamicamente ajustáveis em runtime na classe `SegmentTimeSeriesDataset` (localizada em `src/utilidades.py`):

- **Jittering:** Adição de ruído gaussiano às características temporais.
- **Time-Warping:** Distorção não-linear do vetor tempo via interpolação.
- **Window-Slicing:** Redução da janela e redimensionamento interpolação escalar.
- **MixUp Temporal:** Combinação convexa de características (e Snapping de Rótulos).
- **Decomposição Aditiva:** Adição de ruído apenas ao resíduo isolado da Média Móvel.

---

## 📄 Citação e Licença

Este projeto é disponibilizado sob a Licença MIT. Para publicações acadêmicas que venham a utilizar este código, favor citar o artigo correspondente.

> *As implementações deste repositório foram desenhadas com precisão acadêmica, preservando o estado e os passos lógicos requeridos para assegurar 100% de reproducibilidade dos experimentos documentados.*
