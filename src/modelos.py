"""
Módulo de modelos para o projeto MSR-TCN.
Contém as implementações da Decomposição Adaptativa em Subbandas (ASD),
Redes Convolucionais Temporais (TCN) e a arquitetura completa do MSR-TCN,
juntamente com modelos base (baselines) para comparação.
"""
import warnings
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# Ignorar FutureWarnings relacionadas ao weight_norm em versões mais recentes do PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

class AsymmetricASD1D(nn.Module):
    """
    Decomposição Adaptativa em Subbandas (ASD) 1D Assimétrica.
    Decompõe a série temporal de entrada em 3 subbandas:
    - A Camada 1 gera Alta Frequência (Ruído) e Baixa Frequência.
    - A Camada 2 decompõe a Baixa Frequência em Média Frequência (Sazonalidade) e Baixa Frequência (Tendência).
    """
    def __init__(self, in_channels: int, filter_size: int = 5):
        super(AsymmetricASD1D, self).__init__()
        
        padding = filter_size // 2
        
        # Camada 1
        self.L1_U = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        self.L1_L = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        
        # Camada 2 (Opera apenas na saída L1_L)
        self.L2_U = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        self.L2_L = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        
        # Decimação por 2 após cada filtro
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor):
        # x shape: (batch, in_channels, sequence_length)
        
        # Camada 1
        y1_u = self.pool(self.L1_U(x)) # Alta frequência 1 (Ruído)
        y1_l = self.pool(self.L1_L(x)) # Baixa frequência 1
        
        # Camada 2
        y2_u = self.pool(self.L2_U(y1_l)) # Alta frequência 2 (Sazonalidade)
        y2_l = self.pool(self.L2_L(y1_l)) # Baixa frequência 2 (Tendência)
        
        # Sincronizar as dimensões temporais para as 3 subbandas.
        # y1_u sofreu 1 decimação, y2_u e y2_l sofreram 2 decimações.
        # Max-pool em y1_u para sincronizar as dimensões.
        y1_u_sync = self.pool(y1_u)
        
        return y1_u_sync, y2_u, y2_l


class TemporalBlock(nn.Module):
    """
    Bloco Padrão de Rede Convolucional Temporal (TCN) com convoluções causais dilatadas.
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        self.pad = nn.ConstantPad1d((padding, 0), 0)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.pad, self.conv1, self.relu1, self.dropout1,
                                 self.pad, self.conv2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SubbandTCN(nn.Module):
    """
    TCN para processar uma única subbanda de forma independente.
    """
    def __init__(self, num_inputs: int, num_channels: list = [16, 32], kernel_size: int = 3, dropout: float = 0.2):
        super(SubbandTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, channels, seq_len)
        out = self.network(x)
        out = self.pool(out) # shape: (batch, channels, 1)
        return out.squeeze(-1) # shape: (batch, channels)


class MSRTCN1D(nn.Module):
    """
    MSR-TCN: Rede Convolucional Temporal Residual Multiescala.
    Combina a Decomposição Adaptativa em Subbandas com blocos TCN causais.
    """
    def __init__(self, in_channels: int, seq_len: int, num_classes: int = 3, kernel_size: int = 3, dropout: float = 0.2):
        super(MSRTCN1D, self).__init__()
        self.asd = AsymmetricASD1D(in_channels=in_channels, filter_size=5)
        
        self.tcn_noise = SubbandTCN(in_channels, num_channels=[16, 32], kernel_size=kernel_size, dropout=dropout)
        self.tcn_seasonality = SubbandTCN(in_channels, num_channels=[16, 32], kernel_size=kernel_size, dropout=dropout)
        self.tcn_trend = SubbandTCN(in_channels, num_channels=[16, 32], kernel_size=kernel_size, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(dropout if dropout else 0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        noise, seasonality, trend = self.asd(x)
        
        feat_noise = self.tcn_noise(noise)
        feat_seasonality = self.tcn_seasonality(seasonality)
        feat_trend = self.tcn_trend(trend)
        
        # Concatenar características de todas as subbandas
        combined = torch.cat([feat_noise, feat_seasonality, feat_trend], dim=1)
        out = self.fc(combined)
        return out


class BaselineTCN1D(nn.Module):
    """
    TCN de Controle (Full-Band) para comparação justa.
    A quantidade de canais é igual à soma dos canais das 3 subbandas
    do MSR-TCN, garantindo capacidade de aprendizado equivalente (contagem de parâmetros).
    """
    def __init__(self, in_channels: int, seq_len: int, num_classes: int = 3, kernel_size: int = 3, dropout: float = 0.2):
        super(BaselineTCN1D, self).__init__()
        
        # MSR-TCN usa 3 subbandas com [16, 32] canais = total de [48, 96]
        self.tcn = SubbandTCN(in_channels, num_channels=[48, 96], kernel_size=kernel_size, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(dropout if dropout else 0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        feat = self.tcn(x)
        out = self.fc(feat)
        return out


class SubbandCNN(nn.Module):
    """
    CNN Simples para processar cada subbanda independentemente (Modelo de Ablação).
    """
    def __init__(self, in_channels: int, out_channels: int = 16):
        super(SubbandCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels*2, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool1d(2)
        
        self.flatten = nn.Flatten()
        
    def forward(self, x: torch.Tensor):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.flatten(x)


class MSRCNN1D(nn.Module):
    """
    MSR-CNN (Modelo de Ablação).
    Usa ASD 1D para gerar 3 subbandas e processa cada uma com uma CNN isolada.
    """
    def __init__(self, in_channels: int, seq_len: int, num_classes: int = 3):
        super(MSRCNN1D, self).__init__()
        self.asd = AsymmetricASD1D(in_channels=in_channels, filter_size=5)
        
        self.cnn_noise = SubbandCNN(in_channels, out_channels=16)
        self.cnn_seasonality = SubbandCNN(in_channels, out_channels=16)
        self.cnn_trend = SubbandCNN(in_channels, out_channels=16)
        
        final_seq = max(1, seq_len // 16)
        cnn_out_features = 32 * final_seq 
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_features * 3, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        noise, seasonality, trend = self.asd(x)
        
        feat_noise = self.cnn_noise(noise)
        feat_seasonality = self.cnn_seasonality(seasonality)
        feat_trend = self.cnn_trend(trend)
        
        combined = torch.cat([feat_noise, feat_seasonality, feat_trend], dim=1)
        out = self.fc(combined)
        return out


class BaselineCNN1D(nn.Module):
    """
    CNN Convencional Full-Band (Modelo de Ablação).
    Mesma contagem de parâmetros do MSR-CNN para comparação justa.
    """
    def __init__(self, in_channels: int, seq_len: int, num_classes: int = 3):
        super(BaselineCNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 48, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(48, 96, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool1d(2)
        
        self.pool_extra = nn.MaxPool1d(4)
        self.flatten = nn.Flatten()
        
        final_seq = max(1, seq_len // 16)
            
        self.fc = nn.Sequential(
            nn.Linear(96 * final_seq, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool_extra(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out
