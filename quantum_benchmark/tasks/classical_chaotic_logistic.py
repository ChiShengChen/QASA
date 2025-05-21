import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import csv
import datetime

class Task:
    """
    時序任務範本。
    - 實作資料生成、評估等方法
    - 不含模型本身
    """

    def generate_data(self, total_points=500, r=3.9, x0=0.2):
        """
        產生訓練/測試資料
        Args:
            total_points: 總數據點數
            r: logistic map 參數
            x0: 初始值
        """
        waveform = generate_logistic_map_sequence(total_points, r, x0)
        X_data = torch.from_numpy(waveform[:-1]).unsqueeze(0).unsqueeze(-1)
        Y_data = torch.from_numpy(waveform[1:]).unsqueeze(0).unsqueeze(-1)
        
        train_split_idx = int(total_points * 0.8)
        X_train = X_data[:, :train_split_idx, :]
        Y_train = Y_data[:, :train_split_idx, :]
        X_test_seed = X_data[:, train_split_idx - 20:train_split_idx, :]  # Use last 20 points as seed
        Y_test_true_full = Y_data[:, train_split_idx:, :]
        
        return X_train, Y_train, X_test_seed, Y_test_true_full

    def evaluate(self, y_true, y_pred, verbose=True):
        """
        評估模型預測結果
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        if verbose:
            print(f"MAE: {mae:.6f}")
            print(f"MSE: {mse:.6f}")
        return mae, mse

def generate_logistic_map_sequence(total_points=500, r=3.9, x0=0.2):
    x = np.zeros(total_points)
    x[0] = x0
    for t in range(1, total_points):
        x[t] = r * x[t-1] * (1 - x[t-1])
    return x.astype(np.float32)

def run_classical_chaotic_logistic_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for classical chaotic logistic prediction test")
    seq_len_train = 20
    n_embd_test = 16
    n_head_test = 2
    n_layer_test = 1
    input_dim_test = 1
    output_dim_test = 1
    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test,
        n_layer=n_layer_test,
        block_size=seq_len_train + 10,
        n_intermediate=n_embd_test * 2,
        layer_norm_epsilon=1e-5,
        input_dim=input_dim_test,
        output_dim=output_dim_test
    )
    print(f"Classical Model Config for Chaotic Logistic: {config}\n")
    try:
        model = RWKVModel(config)
        model.to(device)
    except Exception as e:
        print(f"Error instantiating classical RWKVModel: {e}")
        raise
    print("Classical RWKVModel for chaotic logistic instantiated successfully.\n")
    total_points = 500
    r = 3.9
    x0 = 0.2
    waveform = generate_logistic_map_sequence(total_points, r, x0)
    X_data = torch.from_numpy(waveform[:-1]).unsqueeze(0).unsqueeze(-1).to(device)
    Y_data = torch.from_numpy(waveform[1:]).unsqueeze(0).unsqueeze(-1).to(device)
    train_split_idx = int(total_points * 0.8)
    X_train = X_data[:, :train_split_idx, :]
    Y_train = Y_data[:, :train_split_idx, :]
    X_test_seed = X_data[:, train_split_idx - seq_len_train : train_split_idx, :]
    Y_test_true_full = Y_data[:, train_split_idx:, :]
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 1000
    print_every = 50
    num_total_train_points = X_train.shape[1]
    all_epoch_losses = []
    model.train()
    print("Starting classical training for chaotic logistic prediction...")
    training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results_chaotic_logistic_classical"
    os.makedirs(results_dir, exist_ok=True)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_windows_processed = 0
        for i in range(num_total_train_points - seq_len_train + 1):
            optimizer.zero_grad()
            input_window = X_train[:, i : i + seq_len_train, :]
            target_window = Y_train[:, i : i + seq_len_train, :]
            if input_window.shape[1] != seq_len_train:
                continue
            initial_states = None
            predictions, _ = model(input_window, states=initial_states)
            loss = criterion(predictions, target_window)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_windows_processed += 1
        if num_windows_processed > 0:
            average_epoch_loss = epoch_loss / num_windows_processed
            all_epoch_losses.append(average_epoch_loss)
            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_epoch_loss:.6f}")
            if (epoch + 1) % 100 == 0:
                model.eval()
                generated_waveform_points = []
                current_input_sequence = X_test_seed.clone().to(device)
                num_points_to_generate = Y_test_true_full.shape[1]
                param_dtype = next(model.parameters()).dtype
                generation_states = []
                for _ in range(config.n_layer):
                    initial_wkv_aa = torch.zeros(current_input_sequence.size(0), config.n_embd, device=device, dtype=param_dtype)
                    initial_wkv_bb = torch.zeros(current_input_sequence.size(0), config.n_embd, device=device, dtype=param_dtype)
                    initial_wkv_pp = torch.full((current_input_sequence.size(0), config.n_embd), -1e38, device=device, dtype=param_dtype)
                    wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
                    cm_state = torch.zeros(current_input_sequence.size(0), config.n_embd, device=device, dtype=param_dtype)
                    generation_states.append((wkv_state, cm_state))
                with torch.no_grad():
                    for i in range(num_points_to_generate):
                        pred_out, generation_states = model(current_input_sequence, states=generation_states)
                        next_pred_point = pred_out[:, -1, :].clone()
                        generated_waveform_points.append(next_pred_point.squeeze().item())
                        current_input_sequence = torch.cat((current_input_sequence[:, 1:, :], next_pred_point.unsqueeze(1)), dim=1)
                generated_waveform_tensor = torch.tensor(generated_waveform_points, dtype=torch.float32)
                true_waveform_part_for_eval = Y_test_true_full.squeeze().cpu().numpy()
                if len(generated_waveform_tensor) != len(true_waveform_part_for_eval):
                    min_len = min(len(generated_waveform_tensor), len(true_waveform_part_for_eval))
                    true_waveform_part_for_eval = true_waveform_part_for_eval[:min_len]
                    generated_waveform_for_eval = generated_waveform_tensor[:min_len].cpu().numpy()
                else:
                    generated_waveform_for_eval = generated_waveform_tensor.cpu().numpy()
                plt.figure(figsize=(14, 7))
                plt.plot(np.arange(len(true_waveform_part_for_eval)), true_waveform_part_for_eval, label='Ground Truth Chaotic Logistic', color='blue', linestyle='-')
                plt.plot(np.arange(len(generated_waveform_for_eval)), generated_waveform_for_eval, label='Predicted Chaotic Logistic (Classical)', color='green', linestyle='--')
                plt.title(f'Ground Truth vs. Predicted Chaotic Logistic (Classical RWKV) - Epoch {epoch+1}')
                plt.xlabel('Time Step (in test segment)')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_filename = os.path.join(results_dir, f"chaotic_logistic_comparison_classical_{training_start_time}_epoch{epoch+1}.png")
                try:
                    plt.savefig(plot_filename)
                    print(f"Plot saved as {plot_filename}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving plot: {e}")
                model.train()
    print("Classical training finished.\n")
    model.eval()
    print("Starting classical generation for chaotic logistic prediction...")
    generated_waveform_points = []
    current_input_sequence = X_test_seed.clone().to(device)
    num_points_to_generate = Y_test_true_full.shape[1]
    param_dtype = next(model.parameters()).dtype
    generation_states = []
    for _ in range(config.n_layer):
        initial_wkv_aa = torch.zeros(current_input_sequence.size(0), config.n_embd, device=device, dtype=param_dtype)
        initial_wkv_bb = torch.zeros(current_input_sequence.size(0), config.n_embd, device=device, dtype=param_dtype)
        initial_wkv_pp = torch.full((current_input_sequence.size(0), config.n_embd), -1e38, device=device, dtype=param_dtype)
        wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
        cm_state = torch.zeros(current_input_sequence.size(0), config.n_embd, device=device, dtype=param_dtype)
        generation_states.append((wkv_state, cm_state))
    with torch.no_grad():
        for i in range(num_points_to_generate):
            pred_out, generation_states = model(current_input_sequence, states=generation_states)
            next_pred_point = pred_out[:, -1, :].clone()
            generated_waveform_points.append(next_pred_point.squeeze().item())
            current_input_sequence = torch.cat((current_input_sequence[:, 1:, :], next_pred_point.unsqueeze(1)), dim=1)
    generated_waveform_tensor = torch.tensor(generated_waveform_points, dtype=torch.float32)
    true_waveform_part_for_eval = Y_test_true_full.squeeze().cpu().numpy()
    if len(generated_waveform_tensor) != len(true_waveform_part_for_eval):
        min_len = min(len(generated_waveform_tensor), len(true_waveform_part_for_eval))
        true_waveform_part_for_eval = true_waveform_part_for_eval[:min_len]
        generated_waveform_for_eval = generated_waveform_tensor[:min_len].cpu().numpy()
    else:
        generated_waveform_for_eval = generated_waveform_tensor.cpu().numpy()
    mae = mean_absolute_error(true_waveform_part_for_eval, generated_waveform_for_eval)
    mse = mean_squared_error(true_waveform_part_for_eval, generated_waveform_for_eval)
    print(f"Generated chaotic logistic (first 20 points): {generated_waveform_for_eval[:20].tolist()}")
    print(f"True chaotic logistic (first 20 points):      {true_waveform_part_for_eval[:20].tolist()}")
    print(f"Mean Absolute Error (MAE, classical): {mae:.6f}")
    print(f"Mean Squared Error (MSE, classical):  {mse:.6f}\n")
    plot_filename = os.path.join(results_dir, f"chaotic_logistic_comparison_classical_{training_start_time}_final.png")
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(true_waveform_part_for_eval)), true_waveform_part_for_eval, label='Ground Truth Chaotic Logistic', color='blue', linestyle='-')
    plt.plot(np.arange(len(generated_waveform_for_eval)), generated_waveform_for_eval, label='Predicted Chaotic Logistic (Classical)', color='green', linestyle='--')
    plt.title('Ground Truth vs. Predicted Chaotic Logistic (Classical RWKV)')
    plt.xlabel('Time Step (in test segment)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saving plot: {e}")
    csv_filename = os.path.join(results_dir, "model_performance.csv")
    header = [
        'Timestamp', 'Experiment_ID', 'Model_Type', 'Task', 
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth', 
        'learning_rate', 'num_epochs_run', 'seq_len_train',
        'Config_Block_Size', 'Config_n_intermediate', 
        'MAE', 'MSE'
    ]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = f"c_chaotic_logistic_{timestamp}"
    learning_rate = optimizer.param_groups[0]['lr']
    data_row = [
        timestamp, experiment_id, 'Classical', 'ChaoticLogistic',
        config.n_layer, config.n_embd, config.n_head, 'N/A', 'N/A',
        f'{learning_rate:.1e}', num_epochs, seq_len_train,
        config.block_size, config.n_intermediate,
        f'{mae:.6f}', f'{mse:.6f}'
    ]
    file_exists = os.path.isfile(csv_filename)
    is_empty = os.path.getsize(csv_filename) == 0 if file_exists else True
    try:
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or is_empty:
                writer.writerow(header)
            writer.writerow(data_row)
        print(f"Detailed metrics saved to {csv_filename}")
    except Exception as e:
        print(f"Error writing detailed metrics to CSV {csv_filename}: {e}")
    epoch_loss_csv_filename = os.path.join(results_dir, "epoch_losses_classical.csv")
    epoch_loss_header = ["Epoch", "Average Loss"]
    try:
        with open(epoch_loss_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(epoch_loss_header)
            for epoch_num, loss_val in enumerate(all_epoch_losses):
                writer.writerow([epoch_num + 1, f"{loss_val:.6f}" if not np.isnan(loss_val) else "NaN"])
        print(f"Epoch losses saved to {epoch_loss_csv_filename}")
    except Exception as e:
        print(f"Error writing epoch losses to CSV {epoch_loss_csv_filename}: {e}")
    print("\n=== Finished Classical Chaotic Logistic Prediction Test ===\n")

if __name__ == '__main__':
    run_classical_chaotic_logistic_prediction() 