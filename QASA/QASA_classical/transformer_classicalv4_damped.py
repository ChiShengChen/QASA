import os
os.environ['RDMAV_FORK_SAFE'] = '1'

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
import csv
from datetime import datetime

# 保留相同的 CSVLogger
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

# 保留相同的數據生成函數
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ClassicalTransformerV4(nn.Module):
    def __init__(self, sequence_length=50):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = 256
        self.nhead = 4
        self.num_layers = 4  # 使用4層transformer
        
        # 輸入投影和位置編碼
        self.input_proj = nn.Linear(1, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # 4層相同的 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # 輸出層
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, 1]
        x = self.input_proj(x)  # [batch_size, sequence_length, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最後一個時間步
        return self.output_layer(x)

class LitClassicalTransformerV4(pl.LightningModule):
    def __init__(self, sequence_length=50):
        super().__init__()
        self.model = ClassicalTransformerV4(sequence_length=sequence_length)
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()
        
        self.plot_epochs = {1,2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 
                          35, 40, 45, 50, 55, 60, 65, 70, 75, 
                          80, 85, 90, 95, 100}
        self.best_val_loss = float('inf')
        self.plot_data = None
    
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        val_loader = self.trainer.val_dataloaders
        for batch in val_loader:
            self.plot_data = (batch[0].cpu().numpy(), batch[1].cpu().numpy())
            break
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y.squeeze())
        
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
        
        mse = F.mse_loss(y_hat, y.squeeze())
        mae = F.l1_loss(y_hat, y.squeeze())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True)
        
        if loss < self.best_val_loss:
            self.best_val_loss = loss
        
        return loss

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
        
        os.makedirs('classical_v4_plots', exist_ok=True)
        plt.savefig(f'classical_v4_plots/epoch_{epoch}.png')
        plt.close()

    def compare_sorted_unsorted_predictions(self, x, y_true, y_pred, epoch):
        x_last = x[:, -1, 0].reshape(-1)
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x_last, y_true, 'b-', label='True')
        plt.plot(x_last, y_pred, 'r--', label='Predicted')
        plt.title(f'[Unsorted] Epoch {epoch}')
        plt.xlabel('Time (normalized)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

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
        os.makedirs('classical_v4_plots/compare', exist_ok=True)
        plt.savefig(f'classical_v4_plots/compare/epoch_{epoch}_compare.png')
        plt.close()

    def on_train_epoch_end(self):
        if self.current_epoch + 1 in self.plot_epochs and self.plot_data is not None:
            x, y_true = self.plot_data
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                y_pred = self(x_tensor).squeeze().cpu().numpy()
            
            self.plot_predictions(x, y_true, y_pred, self.current_epoch + 1)
            self.compare_sorted_unsorted_predictions(x_tensor.cpu().numpy(), y_true, y_pred, self.current_epoch + 1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=8,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

def main():
    pl.seed_everything(42)
    
    sequence_length = 50
    dataset = generate_damped_oscillator_data(num_points=10000, sequence_length=sequence_length)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=128,
        num_workers=4,
        pin_memory=True
    )
    
    model = LitClassicalTransformerV4(sequence_length=sequence_length)
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/classical_v4',
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
        CSVLogger(save_dir='csv_logs/classical_v4')
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',
        max_epochs=100,
        callbacks=callbacks,
        logger=[
            pl_loggers.TensorBoardLogger(
                save_dir='logs/',
                name='classical_v4',
                version=None,
                default_hp_metric=False
            )
        ],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        deterministic=False,
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed. Best validation loss: {model.best_val_loss:.4f}")

if __name__ == '__main__':
    main() 