import os
import math
import csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import loggers as pl_loggers
import pennylane as qml
from datetime import datetime

# Setup
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
torch.set_num_threads(24)
torch.set_num_interop_threads(24)
torch.backends.cuda.matmul.allow_tf32 = True

# Quantum config
n_qubits = 8
n_layers = 4
dev = qml.device("lightning.qubit", wires=n_qubits + 1)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RX(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)
    for l in range(1, n_layers):
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            qml.RY(weights[l, i], wires=i)
            qml.RZ(weights[l, i], wires=i)
        qml.CNOT(wires=[n_qubits - 1, n_qubits])
        qml.RY(weights[l, -1], wires=n_qubits)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight_shape = (n_layers, n_qubits + 1)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {'weights': self.weight_shape})
        self.input_proj = nn.Linear(input_dim, n_qubits)
        self.bn = nn.BatchNorm1d(n_qubits)
        self.output_proj = nn.Linear(n_qubits, output_dim)

    def forward(self, x, timestep):
        x_proj = torch.tanh(self.input_proj(x))
        x_proj = self.bn(x_proj)
        outputs = [self.qlayer((x_proj[i] + timestep).cpu()).to(x.device) for i in range(x.size(0))]
        out = self.output_proj(torch.stack(outputs))
        return x + out

class QuantumEncoderLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.v_quantum = QuantumLayer(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x_flat = x.reshape(-1, x.size(-1))
        q_out = self.v_quantum(x_flat, torch.tensor(float(x.size(1)), device=x.device))
        q_out = q_out.view(x.size())
        x = self.norm2(x + self.ffn(q_out))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HybridTransformer(nn.Module):
    def __init__(self, sequence_length=50, hidden_dim=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True) for _ in range(num_layers - 1)] +
            [QuantumEncoderLayer(hidden_dim)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.output_layer(x[:, -1])

# Damped oscillator data

def generate_damped_oscillator_data(num_points=5000, sequence_length=50):
    A, gamma, omega, phi = 1.0, 0.1, 2.0, 0
    t = np.linspace(0, 20, num_points)
    x = A * np.exp(-gamma * t) * np.cos(omega * t + phi)
    t = (t - t.mean()) / t.std()
    x = (x - x.mean()) / x.std()
    stride = 1
    n_sequences = (len(t) - sequence_length) // stride
    indices = np.arange(n_sequences) * stride
    X = np.array([t[i:i+sequence_length] for i in indices])
    Y = x[indices + sequence_length]
    return torch.utils.data.TensorDataset(
        torch.FloatTensor(X).reshape(-1, sequence_length, 1),
        torch.FloatTensor(Y).reshape(-1, 1)
    )

def plot_damped_data_sample():
    dataset = generate_damped_oscillator_data(num_points=1000)
    x = dataset.tensors[0][0, :, 0].numpy()
    y = dataset.tensors[1][0].numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x, label='Input Sequence')
    plt.scatter(len(x), y, color='r', label='Target Value')
    plt.title('Damped Oscillator Data Sample')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    os.makedirs('quantum_new_v4/plots', exist_ok=True)
    plt.savefig('quantum_new_v4/plots/data_sample.png')
    plt.close()

# 添加 CSVLogger
class CSVLogger(pl.Callback):
    def __init__(self, save_dir='csv_logs'):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.filename = os.path.join(
            save_dir, 
            f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss',
                'train_mse',
                'train_mae',
                'val_loss',
                'val_mse',
                'val_mae',
                'learning_rate',
                'best_val_loss',
                'best_val_mse',
                'best_val_mae'
            ])
        
        self.metrics = []
        self.best_val_loss = float('inf')
        self.best_val_mse = float('inf')
        self.best_val_mae = float('inf')
    
    def on_train_epoch_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get('val_loss')
        current_val_mse = trainer.callback_metrics.get('val_mse')
        current_val_mae = trainer.callback_metrics.get('val_mae')
        
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        if current_val_mse is not None and current_val_mse < self.best_val_mse:
            self.best_val_mse = current_val_mse
        if current_val_mae is not None and current_val_mae < self.best_val_mae:
            self.best_val_mae = current_val_mae
        
        metrics = {
            'epoch': trainer.current_epoch + 1,
            'train_loss': trainer.callback_metrics.get('train_loss'),
            'train_mse': trainer.callback_metrics.get('train_mse'),
            'train_mae': trainer.callback_metrics.get('train_mae'),
            'val_loss': current_val_loss,
            'val_mse': current_val_mse,
            'val_mae': current_val_mae,
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr'],
            'best_val_loss': self.best_val_loss,
            'best_val_mse': self.best_val_mse,
            'best_val_mae': self.best_val_mae
        }
        
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([metrics[k].item() if isinstance(metrics[k], torch.Tensor) else metrics[k] for k in [
                'epoch',
                'train_loss',
                'train_mse',
                'train_mae',
                'val_loss',
                'val_mse',
                'val_mae',
                'learning_rate',
                'best_val_loss',
                'best_val_mse',
                'best_val_mae'
            ]])

class LitHybridTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = HybridTransformer()
        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.plot_epochs = {1,2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 
                          35, 40, 45, 50, 55, 60, 65, 70, 75, 
                          80, 85, 90, 95, 100}
        self.plot_data = None
        
        # Create directory for plots
        os.makedirs("quantum_new_v4/plots", exist_ok=True)
        os.makedirs("quantum_new_v4/plots/compare", exist_ok=True)

    def on_train_start(self):
        # Store first batch of validation data for consistent plotting
        val_loader = self.trainer.val_dataloaders
        for batch in val_loader:
            self.plot_data = (batch[0].cpu().numpy(), batch[1].cpu().numpy())
            break

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y.squeeze())
        
        # 添加 MSE 和 MAE 的計算
        mse = F.mse_loss(y_hat, y.squeeze())
        mae = F.l1_loss(y_hat, y.squeeze())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y.squeeze())
        
        # 添加 MSE 和 MAE 的計算
        mse = F.mse_loss(y_hat, y.squeeze())
        mae = F.l1_loss(y_hat, y.squeeze())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True)
        
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            torch.save(self.model.state_dict(), 'quantum_new_v4/best_model.pth')
        
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch + 1 in self.plot_epochs and self.plot_data is not None:
            x, y_true = self.plot_data
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                y_pred = self(x_tensor).squeeze().cpu().numpy()
            
            self.plot_predictions(x, y_true, y_pred, self.current_epoch + 1)
            self.compare_sorted_unsorted_predictions(x_tensor.cpu().numpy(), y_true, y_pred, self.current_epoch + 1)

    def plot_predictions(self, x, y_true, y_pred, epoch):
        plt.figure(figsize=(10, 6))
        
        x = x[:, -1, 0].reshape(-1)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y_true = y_true[sort_idx]
        y_pred = y_pred[sort_idx]
        
        plt.plot(x, y_true, 'b-', label='True', alpha=0.5)
        plt.plot(x, y_pred, 'r--', label='Predicted', alpha=0.5)
        plt.title(f'Damped Oscillator - Epoch {epoch}')
        plt.xlabel('Time (normalized)')
        plt.ylabel('Amplitude (normalized)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'quantum_new_v4/plots/epoch_{epoch}.png')
        plt.close()

    def compare_sorted_unsorted_predictions(self, x, y_true, y_pred, epoch):
        x_last = x[:, -1, 0].reshape(-1)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        plt.figure(figsize=(12, 5))

        # Left plot: Original order
        plt.subplot(1, 2, 1)
        plt.plot(x_last, y_true, 'b-', label='True')
        plt.plot(x_last, y_pred, 'r--', label='Predicted')
        plt.title(f'[Unsorted] Epoch {epoch}')
        plt.xlabel('Time (normalized)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # Right plot: Sorted order
        plt.subplot(1, 2, 2)
        sort_idx = np.argsort(x_last)
        x_sorted = x_last[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        plt.plot(x_sorted, y_true_sorted, 'b-', label='True (sorted)')
        plt.plot(x_sorted, y_pred_sorted, 'r--', label='Predicted (sorted)')
        plt.title(f'[Sorted] Epoch {epoch}')
        plt.xlabel('Time (sorted)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'quantum_new_v4/plots/compare/epoch_{epoch}_compare.png')
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

if __name__ == "__main__":
    plot_damped_data_sample()

    sequence_length = 50
    dataset = generate_damped_oscillator_data(num_points=10000, sequence_length=sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128)

    model = LitHybridTransformer()

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/quantum_new_v4',
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min'
        ),
        CSVLogger(save_dir='csv_logs/quantum_new_v4')
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        max_epochs=100,
        callbacks=callbacks,  # 添加 callbacks
        logger=[
            pl_loggers.TensorBoardLogger(
                save_dir='logs/',
                name='quantum_new_v4',
                version=None,
                default_hp_metric=False
            )
        ],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        deterministic=False,
    )

    trainer.fit(model, train_loader, val_loader)
