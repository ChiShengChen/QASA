import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Add imports needed for waveform prediction test
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os # Added for directory creation
import csv # Added for CSV writing
import datetime # Added import

class Task:
    """
    時序任務範本。
    - 實作資料生成、評估等方法
    - 不含模型本身
    """

    def generate_data(self, total_points=500, seq_len_train=9, vocab_size=3):
        """
        產生訓練/測試資料
        Args:
            total_points: 總數據點數
            seq_len_train: 訓練序列長度
            vocab_size: 詞彙表大小
        """
        data_sequence = torch.tensor([i for i in range(vocab_size)] * (seq_len_train // vocab_size + 1), dtype=torch.long)
        X_train = data_sequence[:-1].unsqueeze(0)
        Y_train = data_sequence[1:].unsqueeze(0)
        
        # 生成測試數據
        test_sequence = data_sequence.clone()
        X_test_seed = test_sequence[:-1].unsqueeze(0)
        Y_test_true_full = test_sequence[1:].unsqueeze(0)
        
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

def run_classical_simple_learning_test():
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for classical simple learning test")

    # 1. Define Hyper-small Model Configuration (Classical)
    vocab_size = 3
    seq_len_train = 9 # e.g., 0,1,2,0,1,2,0,1,2
    n_embd_test = 8   # Keep it small, divisible by n_head
    n_head_test = 2
    n_layer_test = 1
    # No quantum parameters needed for classical model
    block_size_test = seq_len_train + 5 # Should be >= any sequence length used

    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test,
        n_layer=n_layer_test,
        vocab_size=vocab_size, # Classical model uses vocab_size
        block_size=block_size_test, 
        n_intermediate=n_embd_test * 2, 
        layer_norm_epsilon=1e-5
        # n_qubits and q_depth are not part of classical ModelConfig
    )
    print(f"Classical Model Config for Learning Test: {config}\n")

    # Instantiate classical model
    try:
        model = RWKVModel(config) # Use classical RWKVModel
        model.to(device) # Move model to device
    except Exception as e:
        print(f"Error instantiating classical RWKVModel: {e}")
        raise
    print("Classical RWKVModel instantiated successfully for learning test.\n")

    # 2. Prepare Training Data (Simple repeating sequence: 0, 1, 2, 0, 1, 2, ...)
    data_sequence = torch.tensor([0, 1, 2] * (seq_len_train // 3 + 1), dtype=torch.long)
    input_sequence = data_sequence[:-1].unsqueeze(0).to(device) # (B=1, T_train-1) -> to device
    target_sequence = data_sequence[1:].unsqueeze(0).to(device)  # (B=1, T_train-1) -> to device

    print(f"Input sequence shape: {input_sequence.shape}, Data: {input_sequence}")
    print(f"Target sequence shape: {target_sequence.shape}, Data: {target_sequence}\n")

    # 3. Training Loop
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 200 
    print_every = 20

    model.train() 
    print("Starting classical training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        initial_states = None 
        logits, updated_states = model(input_sequence, states=initial_states)
        loss = criterion(logits.reshape(-1, vocab_size), target_sequence.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    print("Classical training finished.\n")

    # 4. Prediction/Generation after Training
    model.eval() 
    print("Starting classical generation...")
    
    seed_token = torch.tensor([[0]], dtype=torch.long).to(device) # Start with token 0 -> to device
    generated_sequence = [seed_token.item()]
    num_tokens_to_generate = seq_len_train * 2 
    
    current_input = seed_token
    B_gen = current_input.size(0)
    # device_gen = current_input.device # This will be correctly set now
    dtype_gen = model.wte.weight.dtype 

    generation_states = []
    for _ in range(config.n_layer):
        initial_wkv_aa = torch.zeros(B_gen, config.n_embd, device=device, dtype=dtype_gen)
        initial_wkv_bb = torch.zeros(B_gen, config.n_embd, device=device, dtype=dtype_gen)
        initial_wkv_pp = torch.full((B_gen, config.n_embd), -1e38, device=device, dtype=dtype_gen)
        wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
        cm_state = torch.zeros(B_gen, config.n_embd, device=device, dtype=dtype_gen)
        generation_states.append((wkv_state, cm_state))

    with torch.no_grad():
        for _ in range(num_tokens_to_generate -1):
            logits_gen, generation_states = model(current_input, states=generation_states)
            next_token_logits = logits_gen[:, -1, :]
            predicted_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_sequence.append(predicted_token.item())
            current_input = predicted_token
            if len(generated_sequence) >= num_tokens_to_generate:
                break

    print(f"Original pattern (first {seq_len_train} tokens): {data_sequence[:seq_len_train].tolist()}")
    print(f"Generated sequence (classical): {generated_sequence}")

    expected_pattern_part = data_sequence[:len(generated_sequence)].tolist()
    if generated_sequence == expected_pattern_part:
        print("Classical model seems to have learned the simple periodic pattern!")
    else:
        print("Classical model did not perfectly reproduce the simple periodic pattern.")
        print(f"Expected : {expected_pattern_part}")
        print(f"Generated: {generated_sequence}")
    print("\n=== Finished Classical Simple Sequence Learning Test ===\n")

# --- Test 2: Waveform Prediction (New Function) ---
def run_classical_waveform_prediction_test():
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for classical waveform prediction test")

    print("\n=== Running Classical Waveform Prediction Test ===\n")
    # 1. Define Model Configuration for Classical Waveform Prediction
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
        input_dim=input_dim_test, # For waveform
        output_dim=output_dim_test # For waveform
        # vocab_size should NOT be set for waveform config
    )
    print(f"Classical Model Config for Waveform Test: {config}\n")

    try:
        model = RWKVModel(config) 
        model.to(device) # Move model to device
    except Exception as e:
        print(f"Error instantiating classical RWKVModel for waveform: {e}")
        raise
    print("Classical RWKVModel for waveform instantiated successfully.\n")

    # 2. Prepare Waveform Data (Sine wave)
    total_points = 500
    time_steps = np.linspace(0, 50, total_points)
    waveform = np.sin(time_steps).astype(np.float32)
    
    X_data = torch.from_numpy(waveform[:-1]).unsqueeze(0).unsqueeze(-1).to(device) # -> to device
    Y_data = torch.from_numpy(waveform[1:]).unsqueeze(0).unsqueeze(-1).to(device)   # -> to device

    train_split_idx = int(total_points * 0.8)
    X_train = X_data[:, :train_split_idx, :]
    Y_train = Y_data[:, :train_split_idx, :]
    X_test_seed = X_data[:, train_split_idx - seq_len_train : train_split_idx, :] 
    Y_test_true_full = Y_data[:, train_split_idx:, :] 

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test_seed shape: {X_test_seed.shape}")
    print(f"Y_test_true_full shape: {Y_test_true_full.shape}\n")

    # 3. Training Loop (Sliding Window)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 1000 # Updated
    print_every = 50 # Updated, will also be plot_every
    num_total_train_points = X_train.shape[1]

    all_epoch_losses = [] # To store loss for each epoch

    model.train()
    print("Starting classical training for waveform prediction (with sliding windows)...")
    training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            all_epoch_losses.append(average_epoch_loss) # Store average loss
            if (epoch + 1) % 100 == 0:
                # 存圖
                model.eval()
                current_plot_generated_points_classical = []
                plot_current_input_sequence_classical = X_test_seed.clone().to(device)
                plot_num_points_to_generate_classical = Y_test_true_full.shape[1]
                plot_param_dtype_classical = next(model.parameters()).dtype
                plot_generation_states_classical = []
                for _ in range(config.n_layer):
                    p_initial_wkv_aa_cl = torch.zeros(plot_current_input_sequence_classical.size(0), config.n_embd, device=device, dtype=plot_param_dtype_classical)
                    p_initial_wkv_bb_cl = torch.zeros(plot_current_input_sequence_classical.size(0), config.n_embd, device=device, dtype=plot_param_dtype_classical)
                    p_initial_wkv_pp_cl = torch.full((plot_current_input_sequence_classical.size(0), config.n_embd), -1e38, device=device, dtype=plot_param_dtype_classical)
                    p_wkv_state_cl = (p_initial_wkv_aa_cl, p_initial_wkv_bb_cl, p_initial_wkv_pp_cl)
                    p_cm_state_cl = torch.zeros(plot_current_input_sequence_classical.size(0), config.n_embd, device=device, dtype=plot_param_dtype_classical)
                    plot_generation_states_classical.append((p_wkv_state_cl, p_cm_state_cl))
                with torch.no_grad():
                    for _ in range(plot_num_points_to_generate_classical):
                        plot_pred_out_cl, plot_generation_states_classical = model(plot_current_input_sequence_classical, states=plot_generation_states_classical)
                        plot_next_pred_point_cl = plot_pred_out_cl[:, -1, :].clone()
                        current_plot_generated_points_classical.append(plot_next_pred_point_cl.squeeze().item())
                        plot_current_input_sequence_classical = torch.cat((plot_current_input_sequence_classical[:, 1:, :], plot_next_pred_point_cl.unsqueeze(1)), dim=1)
                plot_generated_waveform_tensor_cl = torch.tensor(current_plot_generated_points_classical, dtype=torch.float32)
                plot_true_waveform_part_for_eval_cl = Y_test_true_full.squeeze().cpu().numpy()
                plot_generated_waveform_for_eval_cl = plot_generated_waveform_tensor_cl.cpu().numpy()
                min_len_plot_cl = min(len(plot_generated_waveform_for_eval_cl), len(plot_true_waveform_part_for_eval_cl))
                plot_generated_waveform_for_eval_cl = plot_generated_waveform_for_eval_cl[:min_len_plot_cl]
                plot_true_waveform_part_for_eval_cl_adj = plot_true_waveform_part_for_eval_cl[:min_len_plot_cl]
                plt.figure(figsize=(14, 7))
                plt.plot(np.arange(len(plot_true_waveform_part_for_eval_cl_adj)), plot_true_waveform_part_for_eval_cl_adj, label='Ground Truth Waveform', color='blue', linestyle='-')
                plt.plot(np.arange(len(plot_generated_waveform_for_eval_cl)), plot_generated_waveform_for_eval_cl, label=f'Predicted Waveform (Classical Epoch {epoch+1})', color='green', linestyle='--')
                plt.title(f'Ground Truth vs. Predicted Waveform (Classical RWKV - Epoch {epoch+1})')
                plt.xlabel('Time Step (in test segment)')
                plt.ylabel('Waveform Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                results_dir_epoch_plot_cl = "results"
                os.makedirs(results_dir_epoch_plot_cl, exist_ok=True)
                plot_filename_epoch_cl = os.path.join(results_dir_epoch_plot_cl, f"waveform_comparison_classical_{training_start_time}_epoch{epoch+1}.png")
                try:
                    plt.savefig(plot_filename_epoch_cl)
                    print(f"Plot saved as {plot_filename_epoch_cl}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving classical epoch plot: {e}")
                model.train()
        elif (epoch + 1) % print_every == 0:
             print(f"Epoch [{epoch+1}/{num_epochs}], No windows processed in this epoch.")
             all_epoch_losses.append(float('nan')) # Store NaN if no windows
    print("Classical training finished.\n")

    # 4. Prediction/Generation and Evaluation (This is the FINAL evaluation)
    model.eval()
    print("Starting classical generation for waveform prediction...")
    
    generated_waveform_points = []
    current_input_sequence = X_test_seed.clone().to(device) # Ensure on device
    num_points_to_generate = Y_test_true_full.shape[1]

    # B_gen = current_input_sequence.size(0)
    # device_gen = current_input_sequence.device # Correctly derived
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

    print(f"Generated Waveform (classical, first 20 points): {generated_waveform_for_eval[:20].tolist()}")
    print(f"True Waveform (classical, first 20 points):      {true_waveform_part_for_eval[:20].tolist()}")
    print(f"Mean Absolute Error (MAE, classical): {mae:.6f}")
    print(f"Mean Squared Error (MSE, classical):  {mse:.6f}\n")

    if mse < 0.1: 
        print("Classical model shows some basic learning on waveform prediction.")
    else:
        print("Classical model did not significantly learn the waveform pattern (MSE > 0.1).")
        
    # --- Result Saving --- 
    results_dir = "results_waveform_classical"
    os.makedirs(results_dir, exist_ok=True)
    
    # 5. Plotting
    plt.figure(figsize=(14, 7))
    plot_time_steps_true = np.arange(len(true_waveform_part_for_eval))
    plot_time_steps_gen = np.arange(len(generated_waveform_for_eval))
    plt.plot(plot_time_steps_true, true_waveform_part_for_eval, label='Ground Truth Waveform', color='blue', linestyle='-')
    plt.plot(plot_time_steps_gen, generated_waveform_for_eval, label='Predicted Waveform (Classical)', color='green', linestyle='--')
    plt.title('Ground Truth vs. Predicted Waveform (Classical RWKV)')
    plt.xlabel('Time Step (in test segment)')
    plt.ylabel('Waveform Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = os.path.join(results_dir, f"waveform_comparison_classical_{training_start_time}_final.png")
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.close() # Close the plot to free memory
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    # --- Enhanced Metrics Saving to CSV (Classical) ---
    csv_filename = os.path.join(results_dir, "model_performance.csv")
    header = [
        'Timestamp', 'Experiment_ID', 'Model_Type', 'Task', 
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth', 
        'learning_rate', 'num_epochs_run', 'seq_len_train',
        'Config_Block_Size', 'Config_n_intermediate', 
        'MAE', 'MSE'
    ]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = f"c_wave_{timestamp}"
    learning_rate = optimizer.param_groups[0]['lr']

    data_row = [
        timestamp, experiment_id, 'Classical', 'Waveform',
        config.n_layer, config.n_embd, config.n_head, 'N/A', 'N/A', # n_qubits, q_depth as N/A
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
        # Consider adding traceback.print_exc() here for more detailed error info

    # --- Save Epoch Losses to CSV (Classical) ---
    epoch_loss_csv_filename_classical = os.path.join(results_dir, "epoch_losses_classical.csv")
    epoch_loss_header_classical = ["Epoch", "Average Loss"]
    try:
        with open(epoch_loss_csv_filename_classical, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(epoch_loss_header_classical)
            for epoch_num, loss_val in enumerate(all_epoch_losses):
                writer.writerow([epoch_num + 1, f"{loss_val:.6f}" if not np.isnan(loss_val) else "NaN"])
        print(f"Epoch losses saved to {epoch_loss_csv_filename_classical}")
    except Exception as e:
        print(f"Error writing classical epoch losses to CSV {epoch_loss_csv_filename_classical}: {e}")
        # Consider adding traceback.print_exc() here if detailed error is needed

    print("\n=== Finished Classical Waveform Prediction Test ===\n")

if __name__ == '__main__':
    # Call both test functions
    run_classical_simple_learning_test()
    run_classical_waveform_prediction_test() 