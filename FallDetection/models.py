import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
class FD_CNNGRU(nn.Module):
    def __init__(self, n_feat=90, n_complex=2, n_classes=2):
        super().__init__()
        self.n_input = n_feat * n_complex
        self.conv1 = nn.Conv1d(self.n_input, 128, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.transpose(1, 2)  # [batch, 180, 2000]
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.transpose(1, 2)  # [batch, time, features]
        out, _ = self.gru(x)   # [batch, seq, 128]
        out = out[:, -1, :]    # last time step
        x = self.fc(out)
        return x
class FD_1DCNN(nn.Module):
    def __init__(self, seq_len=2000, n_feat=90, n_complex=2, n_classes=2):
        super().__init__()
        self.n_input = n_feat * n_complex  # 180

        self.conv1 = nn.Conv1d(self.n_input, 128, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=9, padding=4)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveMaxPool1d(1)  # global pooling

        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: [batch, 2000, 90, 2] → [batch, 2000, 180]
        x = x.view(x.shape[0], x.shape[1], -1)
        # [batch, 2000, 180] → [batch, 180, 2000]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = nn.functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(self.bn3(x))
        x = self.pool3(x)   # [batch, 256, 1]
        x = x.squeeze(-1)   # [batch, 256]
        x = self.fc(x)      # [batch, 2]
        return x
class FD_MLP(nn.Module):
    def __init__(self, n_feat=90, n_complex=2, n_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2000 * n_feat * n_complex, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FD_TransformerE(nn.Module):
    def __init__(self, n_feat=90, n_complex=2, n_classes=2, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(n_feat * n_complex, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)  # [batch, 2000, 180]
        x = self.input_linear(x)                # [batch, 2000, d_model]
        x = self.transformer(x)                 # [batch, 2000, d_model]
        # Pool across time (sequence dimension)
        x = x.transpose(1, 2)                   # [batch, d_model, 2000]
        x = self.pool(x).squeeze(-1)            # [batch, d_model]
        x = self.fc(x)
        return x