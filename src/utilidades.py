"""
Módulo de utilidades para o projeto MSR-TCN.
Contém funções de perda (loss), carregadores de dados e pipelines de geração de dataset
com Aumentação de Dados (Data Augmentation) integrada para séries temporais.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FocalLoss(nn.Module):
    """
    Função Focal Loss para lidar com o desbalanceamento de classes.
    Fornece redimensionamento dinâmico da entropia cruzada com base na confiança da predição.
    """
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(targets.device)[targets]
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()


class SegmentTimeSeriesDataset(Dataset):
    """
    Dataset para carregar dados de séries temporais financeiras segmentadas.
    Suporta Aumentação de Dados em tempo de execução (Jitter, Time-Warp, Window-Slice, MixUp, Decomposition)
    para mitigar o sobreajuste (overfitting) e lidar com o severo desbalanceamento de classes (~90% classe Hold).
    """
    def __init__(self, file_paths: list, window_size: int = 32, is_train: bool = False, 
                 start_date: str = None, end_date: str = None, scaler: dict = None):
        self.X = []
        self.y = []
        self.dates = []
        self.tickers = []
        self.ret_1d = []
        self.is_train = is_train
        self.scaler = scaler if scaler is not None else {}
        
        features_cols = ['Log_Return', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
        
        temp_dfs = []
        for file_path in file_paths:
            if not os.path.exists(file_path): 
                continue
                
            ticker = os.path.basename(file_path).replace('_full.csv', '')
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            if len(df) > window_size:
                # O alvo (target) é o retorno do dia seguinte
                df['Ret_1d'] = df['Close'].pct_change(1).shift(-1) * 100
                temp_dfs.append((df, ticker))
        
        if not temp_dfs:
            self.X, self.y = None, None
            self.da_flags = {}
            return
            
        # Ajusta o padronizador (scaler) usando apenas os dados de treino para evitar vazamento de dados (data leakage)
        if self.is_train:
            combined_df = pd.concat([d[0] for d in temp_dfs])
            for f in features_cols:
                self.scaler[f] = {'mean': combined_df[f].mean(), 'std': combined_df[f].std()}
        
        # Aplica padronização e extrai janelas deslizantes
        for df, ticker in temp_dfs:
            for f in features_cols:
                mean = self.scaler[f]['mean']
                std = self.scaler[f]['std']
                if std != 0:
                    df[f] = (df[f] - mean) / std
                    
            data_X = df[features_cols].values
            data_y = df['Label'].values
            dates_arr = df.index.values
            ret_1d_arr = df['Ret_1d'].fillna(0.0).values
            
            for i in range(len(df) - window_size):
                self.X.append(data_X[i:(i + window_size)])
                self.y.append(data_y[i + window_size])
                
                if not self.is_train:
                    self.dates.append(dates_arr[i + window_size])
                    self.tickers.append(ticker)
                    self.ret_1d.append(ret_1d_arr[i + window_size])
                
        if self.X:
            self.X = np.array(self.X, dtype=np.float32)
            self.y = np.array(self.y, dtype=np.int64)
        else:
            self.X, self.y = None, None
            
        self.da_flags = {}
            
    def __len__(self):
        return len(self.X) if self.X is not None else 0
        
    def set_da_flags(self, flags: dict):
        """Habilita técnicas específicas de Aumentação de Dados."""
        self.da_flags = flags

    def apply_data_augmentation(self, x: np.ndarray, y: int):
        """Aplica as estratégias de aumentação configuradas à janela de amostra."""
        if self.da_flags.get('use_jitter'):
            x += np.random.normal(0, 0.03, x.shape)
            
        if self.da_flags.get('use_warp'):
            warp_factor = np.zeros_like(x)
            for c in range(x.shape[1]):
                warp_factor[:, c] = np.interp(
                    np.linspace(0, 1, len(x)), 
                    [0, 0.5, 1], 
                    np.random.normal(1, 0.1, 3)
                )
            x *= warp_factor
            
        if self.da_flags.get('use_slice'):
            slice_len = int(len(x) * 0.9)
            start_idx = np.random.randint(0, len(x) - slice_len)
            sliced_x = np.zeros_like(x)
            for c in range(x.shape[1]):
                sliced_x[:, c] = np.interp(
                    np.linspace(0, 1, len(x)), 
                    np.linspace(0, 1, slice_len), 
                    x[start_idx : start_idx + slice_len, c]
                )
            x = sliced_x
            
        if self.da_flags.get('use_mixup'):
            idx2 = np.random.randint(0, len(self.X))
            x2 = self.X[idx2].copy()
            lam = np.random.beta(0.5, 0.5)
            x = lam * x + (1 - lam) * x2
            # Label snapping: mantém o rótulo da amostra dominante
            if lam < 0.5: 
                y = self.y[idx2] 
                
        if self.da_flags.get('use_decomp'):
            # Decomposição Temporal: adiciona ruído ao resíduo, preserva a tendência de média móvel
            trend = np.zeros_like(x)
            for c in range(x.shape[1]):
                trend[:, c] = np.convolve(x[:, c], np.ones(3)/3, mode='same')
            x = trend + (x - trend) + np.random.normal(0, 0.05, x.shape)
            
        return x, y

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        
        if self.is_train:
            x, y = self.apply_data_augmentation(x, y)
                
        x_tensor = torch.tensor(x, dtype=torch.float32).transpose(0, 1)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor
