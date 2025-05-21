import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class Task:
    """
    時序任務範本。
    - 實作資料生成、評估等方法
    - 不含模型本身
    """

    def generate_data(self, total_points=500, seq_len_train=20, train_split=0.8, a=0.05, w=1.0, seed=42):
        """
        產生訓練/測試資料 (Adapted from classical_damped_oscillation.py)
        Args:
            total_points: 總數據點數
            seq_len_train: 訓練序列長度 (used for X_test_seed length)
            train_split: 訓練集比例
            a: Damping factor
            w: Frequency
            seed: Random seed for numpy
        """
        np.random.seed(seed)
        t = np.linspace(0, 50, total_points)
        waveform = np.exp(-a * t) * np.sin(w * t)
        waveform = waveform.astype(np.float32)

        # Ensure waveform has enough points for X and Y data
        if len(waveform) < 2:
            print(f"Warning: Generated waveform is too short (length {len(waveform)}) for quantum_damped_oscillation. Adjusting total_points or parameters may be needed.")
            # Fallback to a simple default if waveform is too short
            t = np.linspace(0, 1, 2)
            waveform = np.sin(t).astype(np.float32)
            if len(waveform) < 2: # Should not happen with fallback
                 waveform = np.array([0.0, 0.1], dtype=np.float32) 

        X_data = torch.from_numpy(waveform[:-1]).unsqueeze(0).unsqueeze(-1) # (B=1, Total-1, Dim=1)
        Y_data = torch.from_numpy(waveform[1:]).unsqueeze(0).unsqueeze(-1)   # (B=1, Total-1, Dim=1)
        
        train_split_idx = int(len(waveform[:-1]) * train_split) # Split based on actual data points available for X

        # Ensure train_split_idx is valid
        if train_split_idx <= 0 and len(waveform[:-1]) > 0 : train_split_idx = 1 # at least one training sample if data exists
        if train_split_idx > len(waveform[:-1]): train_split_idx = len(waveform[:-1])

        X_train = X_data[:, :train_split_idx, :]
        Y_train = Y_data[:, :train_split_idx, :]

        # Ensure X_test_seed can be formed
        seed_start_idx = train_split_idx - seq_len_train
        if seed_start_idx < 0: seed_start_idx = 0
        
        X_test_seed = X_data[:, seed_start_idx : train_split_idx, :] 
        
        # If X_test_seed becomes empty due to splitting/seq_len_train, create a minimal seed
        if X_test_seed.shape[1] == 0 and X_train.shape[1] > 0:
            X_test_seed = X_train[:, -1:, :] # Use last point of training data as seed
        elif X_test_seed.shape[1] == 0 and X_train.shape[1] == 0: # No training data either
             X_test_seed = torch.zeros((1,1,1), dtype=torch.float32) # Default single zero seed

        Y_test_true_full = Y_data[:, train_split_idx:, :]
        
        # If Y_test_true_full is empty, but there should be test data
        if Y_test_true_full.shape[1] == 0 and train_split_idx < len(waveform[1:]):
            # This might happen if train_split is 1.0 or very close
            # Provide at least one point if possible for Y_test_true_full if Y_data has more points
            if len(waveform[1:]) > train_split_idx:
                 Y_test_true_full = Y_data[:, train_split_idx:train_split_idx+1, :]
            else: # Fallback: no test points available
                 Y_test_true_full = torch.zeros((1,0,1), dtype=torch.float32) # Empty test tensor

        return X_train, Y_train, X_test_seed, Y_test_true_full

    def evaluate(self, y_true, y_pred, verbose=True):
        """
        評估模型預測結果 (Adapted from classical_damped_oscillation.py)
        Calculates Mean Absolute Error (MAE) and Mean Squared Error (MSE).
        Args:
            y_true: Ground truth values (NumPy array).
            y_pred: Predicted values (NumPy array).
            verbose: If True, prints the MAE and MSE.
        Returns:
            A tuple (mae, mse).
        """
        # Ensure inputs are numpy arrays
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        if y_true.shape != y_pred.shape:
            print(f"    Task Evaluate Warning: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}. Errors might be inaccurate.")
            min_len = min(len(y_true), len(y_pred))
            if min_len == 0:
                 if verbose:
                      print(f"    Mean Absolute Error (MAE): inf")
                      print(f"    Mean Squared Error (MSE): inf")
                 return np.inf, np.inf
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        if y_true.size == 0:
            if verbose:
                print(f"    Mean Absolute Error (MAE): 0.000000 (empty inputs)")
                print(f"    Mean Squared Error (MSE): 0.000000 (empty inputs)")
            return 0.0, 0.0

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        if verbose:
            print(f"    Mean Absolute Error (MAE): {mae:.6f}")
            print(f"    Mean Squared Error (MSE): {mse:.6f}") # Removed extra \n for consistency
        return mae, mse

def run_quantum_damped_oscillation_prediction():
    print("\n=== Running Quantum Damped Oscillation Prediction Test ===\n")
    # Model config
    seq_len_train = 20
    n_embd_test = 16
    n_head_test = 2
    n_layer_test = 1
    n_qubits_test = 4
    q_depth_test = 1
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
        output_dim=output_dim_test,
        n_qubits=n_qubits_test,
        q_depth=q_depth_test
    )
    print(f"Quantum Model Config for Damped Oscillation: {config}\n")
    try:
        model = QuantumRWKVModel(config)
    except Exception as e:
        print(f"Error instantiating QuantumRWKVModel for damped oscillation: {e}")
        raise
    print("QuantumRWKVModel for damped oscillation instantiated successfully.\n")

    # Damped oscillation: y = exp(-a t) * sin(w t)
    total_points = 500
    t = np.linspace(0, 50, total_points)
    a = 0.05  # Damping factor
    w = 1.0   # Frequency
    waveform = np.exp(-a * t) * np.sin(w * t)
    waveform = waveform.astype(np.float32)

    X_data = torch.from_numpy(waveform[:-1]).unsqueeze(0).unsqueeze(-1)
    Y_data = torch.from_numpy(waveform[1:]).unsqueeze(0).unsqueeze(-1)

    train_split_idx = int(total_points * 0.8)
    X_train = X_data[:, :train_split_idx, :]
    Y_train = Y_data[:, :train_split_idx, :]
    X_test_seed = X_data[:, train_split_idx - seq_len_train : train_split_idx, :]
    Y_test_true_full = Y_data[:, train_split_idx:, :]

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test_seed shape: {X_test_seed.shape}")
    print(f"Y_test_true_full shape: {Y_test_true_full.shape}\n")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 1000
    print_every = 50
    num_total_train_points = X_train.shape[1]
    all_epoch_losses = []

    training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results_damped_oscillation_quantum"
    os.makedirs(results_dir, exist_ok=True)
    print("Starting quantum training for damped oscillation prediction...")
    model.train()
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
                plt.plot(np.arange(len(true_waveform_part_for_eval)), true_waveform_part_for_eval, label='Ground Truth Damped Oscillation', color='blue', linestyle='-')
                plt.plot(np.arange(len(generated_waveform_for_eval)), generated_waveform_for_eval, label='Predicted Damped Oscillation (Quantum)', color='red', linestyle='--')
                plt.title(f'Ground Truth vs. Predicted Damped Oscillation (Quantum RWKV) - Epoch {epoch+1}')
                plt.xlabel('Time Step (in test segment)')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_filename = os.path.join(results_dir, f"damped_oscillation_comparison_quantum_{training_start_time}_epoch{epoch+1}.png")
                try:
                    plt.savefig(plot_filename)
                    print(f"Plot saved as {plot_filename}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving plot: {e}")
                model.train()
    print("Quantum training finished.\n")

    # Generation and evaluation
    model.eval()
    print("Starting quantum generation for damped oscillation prediction...")
    generated_waveform_points = []
    current_input_sequence = X_test_seed.clone()
    num_points_to_generate = Y_test_true_full.shape[1]
    param_dtype = next(model.parameters()).dtype
    generation_states = []
    for _ in range(config.n_layer):
        initial_wkv_aa = torch.zeros(current_input_sequence.size(0), config.n_embd, dtype=param_dtype)
        initial_wkv_bb = torch.zeros(current_input_sequence.size(0), config.n_embd, dtype=param_dtype)
        initial_wkv_pp = torch.full((current_input_sequence.size(0), config.n_embd), -1e38, dtype=param_dtype)
        wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
        cm_state = torch.zeros(current_input_sequence.size(0), config.n_embd, dtype=param_dtype)
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
    mae, mse = self.evaluate(true_waveform_part_for_eval, generated_waveform_for_eval)
    print(f"Generated Damped Oscillation (first 20 points): {generated_waveform_for_eval[:20].tolist()}")
    print(f"True Damped Oscillation (first 20 points):      {true_waveform_part_for_eval[:20].tolist()}")
    print(f"Mean Absolute Error (MAE, quantum): {mae:.6f}")
    print(f"Mean Squared Error (MSE, quantum):  {mse:.6f}\n")
    results_dir = "results_damped_oscillation_quantum"
    os.makedirs(results_dir, exist_ok=True)
    plot_filename = os.path.join(results_dir, f"damped_oscillation_comparison_quantum_{training_start_time}_final.png")
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(true_waveform_part_for_eval)), true_waveform_part_for_eval, label='Ground Truth Damped Oscillation', color='blue', linestyle='-')
    plt.plot(np.arange(len(generated_waveform_for_eval)), generated_waveform_for_eval, label='Predicted Damped Oscillation (Quantum)', color='red', linestyle='--')
    plt.title('Ground Truth vs. Predicted Damped Oscillation (Quantum RWKV)')
    plt.xlabel('Time Step (in test segment)')
    plt.ylabel('Oscillation Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error saving plot: {e}")
    # Save metrics to CSV
    csv_filename = os.path.join(results_dir, "model_performance.csv")
    header = [
        'Timestamp', 'Experiment_ID', 'Model_Type', 'Task', 
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth', 
        'learning_rate', 'num_epochs_run', 'seq_len_train',
        'Config_Block_Size', 'Config_n_intermediate', 
        'MAE', 'MSE'
    ]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = f"q_damped_{timestamp}"
    learning_rate = optimizer.param_groups[0]['lr']
    data_row = [
        timestamp, experiment_id, 'Quantum', 'DampedOscillation',
        config.n_layer, config.n_embd, config.n_head, config.n_qubits, config.q_depth,
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
    # Save epoch losses
    epoch_loss_csv_filename = os.path.join(results_dir, "epoch_losses_quantum.csv")
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
    print("\n=== Finished Quantum Damped Oscillation Prediction Test ===\n")

if __name__ == '__main__':
    run_quantum_damped_oscillation_prediction() 