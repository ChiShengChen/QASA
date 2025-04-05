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

# CSVLogger 和 generate_damped_oscillator_data 函數保持不變
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
        
        # 更新最佳值
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
            writer.writerow([
                metrics['epoch'],
                metrics['train_loss'].item() if metrics['train_loss'] is not None else None,
                metrics['train_mse'].item() if metrics['train_mse'] is not None else None,
                metrics['train_mae'].item() if metrics['train_mae'] is not None else None,
                metrics['val_loss'].item() if metrics['val_loss'] is not None else None,
                metrics['val_mse'].item() if metrics['val_mse'] is not None else None,
                metrics['val_mae'].item() if metrics['val_mae'] is not None else None,
                metrics['learning_rate'],
                metrics['best_val_loss'].item() if isinstance(metrics['best_val_loss'], torch.Tensor) else metrics['best_val_loss'],
                metrics['best_val_mse'].item() if isinstance(metrics['best_val_mse'], torch.Tensor) else metrics['best_val_mse'],
                metrics['best_val_mae'].item() if isinstance(metrics['best_val_mae'], torch.Tensor) else metrics['best_val_mae']
            ])
        
        self.metrics.append(metrics)

def generate_damped_oscillator_data(num_points=5000, sequence_length=50):
    # 參數設置
    A = 1.0        # 振幅
    gamma = 0.1    # 阻尼係數
    omega = 2.0    # 角頻率
    phi = 0        # 初始相位
    
    # 生成時間序列
    t = np.linspace(0, 20, num_points)
    
    # 計算阻尼諧振子位置
    x = A * np.exp(-gamma * t) * np.cos(omega * t + phi)
    
    # 標準化數據
    t = (t - t.mean()) / t.std()
    x = (x - x.mean()) / x.std()
    
    # 創建時序序列
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

class TransformerDamped(nn.Module):
    def __init__(self, sequence_length=50):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = 1
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        
        # 輸入投影和位置編碼
        self.input_proj = nn.Linear(1, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer Encoder
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
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, 1]
        batch_size = x.shape[0]
        
        # 輸入投影
        x = self.input_proj(x)  # [batch_size, sequence_length, d_model]
        
        # 位置編碼
        x = self.pos_encoder(x)
        
        # Transformer 編碼
        x = self.transformer_encoder(x)  # [batch_size, sequence_length, d_model]
        
        # 取最後一個時間步的輸出
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # 輸出層
        x = self.output_layer(x)  # [batch_size, 1]
        
        return x

class LitTransformerDamped(pl.LightningModule):
    def __init__(self, sequence_length=50):
        super().__init__()
        self.model = TransformerDamped(sequence_length=sequence_length)
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()
        
        self.plot_epochs = {1, 15, 30, 60, 75, 100}
        self.best_val_loss = float('inf')
        self.plot_data = None
    
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        val_batch = next(iter(self.trainer.val_dataloaders))
        self.plot_data = (
            val_batch[0].detach().cpu().numpy(),
            val_batch[1].detach().cpu().numpy()
        )
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.view(-1)
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y)
        
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.view(-1)
        y = y.view(-1)
        val_loss = self.loss_fn(y_hat, y)
        
        val_mse = F.mse_loss(y_hat, y)
        val_mae = F.l1_loss(y_hat, y)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', val_mse, on_epoch=True)
        self.log('val_mae', val_mae, on_epoch=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.0001,  # 使用更小的學習率
            weight_decay=0.01,
            betas=(0.9, 0.999)
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
        
        os.makedirs('transformer_damped_plots', exist_ok=True)
        plt.savefig(f'transformer_damped_plots/epoch_{epoch}.png')
        plt.close()

    def on_validation_epoch_end(self):
        current_val_loss = self.trainer.callback_metrics.get('val_loss')
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            
            # 保存最佳模型
            model_path = os.path.join('models', 'transformer_damped', 'best_model.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'val_loss': self.best_val_loss,
            }, model_path)
    
    def on_train_epoch_end(self):
        if self.current_epoch + 1 in self.plot_epochs and self.plot_data is not None:
            # 使用固定的驗證集數據進行預測和繪圖
            x, y_true = self.plot_data
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                y_pred = self(x_tensor).cpu().numpy()
            
            self.plot_predictions(x, y_true, y_pred, self.current_epoch + 1)
            
        # 每個 epoch 結束時保存檢查點
        ckpt_path = os.path.join('checkpoints', 'transformer_damped', f'epoch_{self.current_epoch+1}.ckpt')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        self.trainer.save_checkpoint(ckpt_path)

def plot_damped_data_sample():
    # 生成數據
    dataset = generate_damped_oscillator_data(num_points=1000)
    
    # 獲取時間和位置數據
    x = dataset.tensors[0][0, :, 0].numpy()  # 取第一個序列
    y = dataset.tensors[1][0].numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, label='Input Sequence')
    plt.scatter(len(x), y, color='r', label='Target Value')
    plt.title('Damped Oscillator Data Sample')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('transformer_damped_plots', exist_ok=True)
    plt.savefig('transformer_damped_plots/data_sample.png')
    plt.close()

def main():
    pl.seed_everything(42)
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        print("No GPU available, using CPU")
    
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
    
    model = LitTransformerDamped(sequence_length=sequence_length)
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/transformer_damped',
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
        CSVLogger(save_dir='csv_logs/transformer_damped')
    ]
    
    trainer = pl.Trainer(
        accelerator='cuda',
        devices=1,
        precision='16-mixed',
        max_epochs=100,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=[
            pl_loggers.TensorBoardLogger(
                save_dir='logs/',
                name='transformer_damped',
                version=None,
                default_hp_metric=False
            )
        ],
        log_every_n_steps=25,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        deterministic=False,
    )
    
    # 在 main 函數開始時添加
    plot_damped_data_sample()
    
    trainer.fit(model, train_loader, val_loader)
    
    final_model_path = os.path.join('models', 'transformer_damped', 'final_model.pth')
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        'epoch': trainer.current_epoch,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        'val_loss': trainer.callback_metrics.get('val_loss'),
    }, final_model_path)
    
    print(f"Training completed. Best validation loss: {model.best_val_loss:.4f}")

if __name__ == '__main__':
    main() 