import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt # Added for plotting
import os
import datetime

class Task:
    """
    時序任務範本。
    - 實作資料生成、評估等方法
    - 不含模型本身
    """

    def generate_data(self, total_points=500, seq_len_train=20, train_split=0.8, seed=42):
        np.random.seed(seed)
        time_steps = np.linspace(0, 50, total_points)
        waveform = np.sin(time_steps).astype(np.float32)
        X_data = torch.from_numpy(waveform[:-1]).unsqueeze(0).unsqueeze(-1)
        Y_data = torch.from_numpy(waveform[1:]).unsqueeze(0).unsqueeze(-1)
        train_split_idx = int(total_points * train_split)
        X_train = X_data[:, :train_split_idx, :]
        Y_train = Y_data[:, :train_split_idx, :]
        X_test_seed = X_data[:, train_split_idx - seq_len_train : train_split_idx, :]
        Y_test_true_full = Y_data[:, train_split_idx:, :]
        return X_train, Y_train, X_test_seed, Y_test_true_full

    def evaluate(self, y_true, y_pred, verbose=True):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        if verbose:
            print(f"Mean Absolute Error (MAE): {mae:.6f}")
            print(f"Mean Squared Error (MSE): {mse:.6f}\n")
        return mae, mse

def run_waveform_prediction_test():
    # 1. Define Model Configuration for Waveform Prediction
    seq_len_train = 20 # Length of input sequence window for training
    n_embd_test = 16   # Must be divisible by n_head
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
    print(f"Quantum Model Config for Waveform Test: {config}\n")

    # Instantiate model
    try:
        model = QuantumRWKVModel(config)
    except Exception as e:
        print(f"Error instantiating QuantumRWKVModel for waveform: {e}")
        raise
    print("QuantumRWKVModel for waveform instantiated successfully.\n")

    # 2. Prepare Waveform Data (Sine wave)
    total_points = 500
    time_steps = np.linspace(0, 50, total_points)
    waveform = np.sin(time_steps).astype(np.float32)
    
    # Create input sequences and target sequences
    # Input: waveform[t] -> Target: waveform[t+1]
    # We'll feed sequences of length `seq_len_train` and predict the next value for each point in the sequence.
    X_data = torch.from_numpy(waveform[:-1]).unsqueeze(0).unsqueeze(-1) # (B=1, Total_points-1, 1)
    Y_data = torch.from_numpy(waveform[1:]).unsqueeze(0).unsqueeze(-1)   # (B=1, Total_points-1, 1)

    # Split into training and test (simple split for now)
    train_split_idx = int(total_points * 0.8)
    X_train = X_data[:, :train_split_idx, :]
    Y_train = Y_data[:, :train_split_idx, :]
    X_test_seed = X_data[:, train_split_idx - seq_len_train : train_split_idx, :] # Seed for generation
    Y_test_true_full = Y_data[:, train_split_idx:, :] # Full true sequence for comparison

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test_seed shape: {X_test_seed.shape}")
    print(f"Y_test_true_full shape: {Y_test_true_full.shape}\n")

    # 3. Training Loop
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 100 # Adjust as needed
    print_every = 10
    batch_size_train = 1 # Using the whole sequence as one batch for simplicity

    model.train()
    print("Starting training for waveform prediction (with sliding windows)...")
    num_total_train_points = X_train.shape[1]

    training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results_waveform_quantum_simple"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_windows_processed = 0
        
        # States should be managed across windows if we want to capture longer dependencies
        # For simplicity in this iteration, we reset states for each window (initial_states = None)
        # More advanced: Initialize states at epoch start, pass and update through windows.
        # Let's try resetting states for each window first.

        for i in range(num_total_train_points - seq_len_train + 1):
            optimizer.zero_grad()
            
            input_window = X_train[:, i : i + seq_len_train, :] 
            target_window = Y_train[:, i : i + seq_len_train, :] 

            if input_window.shape[1] != seq_len_train: # Should not happen with correct loop bounds
                continue

            initial_states = None # Reset states for each window for simplicity
            predictions, _ = model(input_window, states=initial_states)
            
            loss = criterion(predictions, target_window)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_windows_processed += 1
        
        if num_windows_processed > 0:
            average_epoch_loss = epoch_loss / num_windows_processed
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
                plot_time_steps_true = np.arange(len(true_waveform_part_for_eval))
                plot_time_steps_gen = np.arange(len(generated_waveform_for_eval))
                plt.plot(plot_time_steps_true, true_waveform_part_for_eval, label='Ground Truth Waveform', color='blue', linestyle='-')
                plt.plot(plot_time_steps_gen, generated_waveform_for_eval, label='Predicted Waveform (Quantum)', color='red', linestyle='--')
                plt.title(f'Ground Truth vs. Predicted Waveform (Quantum RWKV) - Epoch {epoch+1}')
                plt.xlabel('Time Step (in test segment)')
                plt.ylabel('Waveform Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_filename = os.path.join(results_dir, f"waveform_prediction_comparison_quantum_rwkv_{training_start_time}_epoch{epoch+1}.png")
                try:
                    plt.savefig(plot_filename)
                    print(f"Plot saved as {plot_filename}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving or showing plot: {e}")
                model.train()
        elif (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], No windows processed in this epoch.")

    print("Training finished.\n")

    # 4. Prediction/Generation and Evaluation after Training
    model.eval()
    print("Starting generation for waveform prediction...")
    
    generated_waveform_points = []
    current_input_sequence = X_test_seed.clone() # (1, seq_len_train, 1)
    num_points_to_generate = Y_test_true_full.shape[1]

    # Initialize generation states
    B_gen = current_input_sequence.size(0)
    device_gen = current_input_sequence.device
    param_dtype = next(model.parameters()).dtype
    generation_states = []
    for _ in range(config.n_layer):
        initial_wkv_aa = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=param_dtype)
        initial_wkv_bb = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=param_dtype)
        initial_wkv_pp = torch.full((B_gen, config.n_embd), -1e38, device=device_gen, dtype=param_dtype)
        wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
        cm_state = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=param_dtype)
        generation_states.append((wkv_state, cm_state))

    with torch.no_grad():
        for i in range(num_points_to_generate):
            # Model expects (B, T, C) where T is sequence length
            # For autoregressive, T=1 for the newest point, but RWKV processes sequences.
            # We feed the current window of `seq_len_train` points.
            pred_out, generation_states = model(current_input_sequence, states=generation_states)
            
            # The prediction for the *next* point is the output for the last element in the input sequence
            next_pred_point = pred_out[:, -1, :].clone() # (B=1, 1, OutputDim=1)
            generated_waveform_points.append(next_pred_point.squeeze().item())
            
            # Update current_input_sequence: roll window and append prediction
            # (1, seq_len_train, 1)
            current_input_sequence = torch.cat((current_input_sequence[:, 1:, :], next_pred_point.unsqueeze(1)), dim=1)

    generated_waveform_tensor = torch.tensor(generated_waveform_points, dtype=torch.float32)
    # Ensure Y_test_true_full is squeezed correctly for comparison
    true_waveform_part_for_eval = Y_test_true_full.squeeze().cpu().numpy() 
    # Ensure generated_waveform_tensor matches length of true_waveform_part_for_eval for metrics
    if len(generated_waveform_tensor) != len(true_waveform_part_for_eval):
        print(f"Warning: Length mismatch. Generated: {len(generated_waveform_tensor)}, True: {len(true_waveform_part_for_eval)}")
        # Truncate or pad if necessary, or re-evaluate generation length logic.
        # For now, let's use the shorter length for metrics if they mismatch due to generation stopping early.
        min_len = min(len(generated_waveform_tensor), len(true_waveform_part_for_eval))
        true_waveform_part_for_eval = true_waveform_part_for_eval[:min_len]
        generated_waveform_for_eval = generated_waveform_tensor[:min_len].cpu().numpy()
    else:
        generated_waveform_for_eval = generated_waveform_tensor.cpu().numpy()

    mae = mean_absolute_error(true_waveform_part_for_eval, generated_waveform_for_eval)
    mse = mean_squared_error(true_waveform_part_for_eval, generated_waveform_for_eval)

    print(f"Generated Waveform (first 20 points): {generated_waveform_for_eval[:20].tolist()}")
    print(f"True Waveform (first 20 points):      {true_waveform_part_for_eval[:20].tolist()}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE):  {mse:.6f}\n")

    if mse < 0.1: # Arbitrary threshold for basic learning
        print("Model shows some basic learning on waveform prediction.")
    else:
        print("Model did not significantly learn the waveform pattern (MSE > 0.1).")

    # 5. Plotting Ground Truth vs. Predicted Waveform
    plt.figure(figsize=(14, 7))
    plot_time_steps_true = np.arange(len(true_waveform_part_for_eval))
    plot_time_steps_gen = np.arange(len(generated_waveform_for_eval)) # Should be same length now
    
    plt.plot(plot_time_steps_true, true_waveform_part_for_eval, label='Ground Truth Waveform', color='blue', linestyle='-')
    plt.plot(plot_time_steps_gen, generated_waveform_for_eval, label='Predicted Waveform', color='red', linestyle='--')
    
    plt.title('Ground Truth vs. Predicted Waveform (Quantum RWKV)')
    plt.xlabel('Time Step (in test segment)')
    plt.ylabel('Waveform Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show the plot
    plot_filename = os.path.join(results_dir, f"waveform_prediction_comparison_quantum_rwkv_{training_start_time}_final.png")
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    except Exception as e:
        print(f"Error saving or showing plot: {e}")
        print("Ensure matplotlib is installed correctly and a suitable backend is available if using plt.show().")

if __name__ == '__main__':
    run_waveform_prediction_test() 