import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import traceback
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class Task:
    """
    時序任務範本。
    - 實作資料生成、評估等方法
    - 不含模型本身
    """

    # Add vocab_size as a potential attribute for auto-detection in run_all_benchmarks
    vocab_size = 3 # Default, can be overridden by generate_data call

    def generate_data(self, total_points=500, seq_len_train=9, vocab_size=3):
        """
        產生訓練/測試資料 (Adapted from classical_learning.py)
        Args:
            total_points: 總數據點數 (currently unused in this simple generation)
            seq_len_train: 訓練序列長度
            vocab_size: 詞彙表大小
        """
        self.vocab_size = vocab_size # Store for potential later use or inspection

        # Create a repeating sequence: 0, 1, 2, 0, 1, 2, ... up to vocab_size
        # Ensure sequence is long enough for seq_len_train
        cycle_len = vocab_size
        num_cycles = (seq_len_train + cycle_len -1) // cycle_len # Ensure enough tokens for seq_len_train
        
        base_pattern = torch.tensor([i for i in range(vocab_size)], dtype=torch.long)
        data_sequence = base_pattern.repeat(num_cycles)[:seq_len_train + 1] # Create sequence of len seq_len_train + 1

        if len(data_sequence) <= 1:
             # Handle edge case: if seq_len_train is 0 or vocab_size makes data_sequence too short
             # Fallback to a minimal sequence if seq_len_train is very small
             print(f"Warning: Generated data_sequence is too short (length {len(data_sequence)}) for quantum_learning with seq_len_train={seq_len_train}, vocab_size={vocab_size}. Adjusting.")
             if vocab_size == 0 : vocab_size = 1 # avoid division by zero if vocab_size is 0
             data_sequence = torch.tensor([i for i in range(vocab_size)] * 2, dtype=torch.long) # e.g., [0,1,0,1]
             if len(data_sequence) <=1 : # Still too short (e.g. vocab_size = 1)
                  data_sequence = torch.tensor([0,0], dtype=torch.long)


        X_train = data_sequence[:-1].unsqueeze(0)  # (1, seq_len_train)
        Y_train = data_sequence[1:].unsqueeze(0)   # (1, seq_len_train)
        
        # For testing, use a similar seed but could be different in more complex tasks
        # Here, test data is essentially predicting the continuation of the same pattern
        # X_test_seed will be the first part of Y_test_true_full
        # Y_test_true_full will be the sequence to predict
        
        # Let's make test seed shorter for generation, e.g., first token
        # And Y_test_true_full the next N tokens (e.g., seq_len_train tokens)
        
        if X_train.shape[1] > 0:
            X_test_seed = X_train[:, :1] # Seed with the first token of X_train, shape (1,1)
        else: # If X_train is empty, create a default seed
            X_test_seed = torch.tensor([[0]], dtype=torch.long)


        # Y_test_true_full should be the sequence we want to predict after the seed
        # Let's try to predict a sequence of length `seq_len_train`
        # The actual Y_test_true_full should be based on the task's true continuation
        # For this simple repeating task, it's just the next elements of the pattern.
        
        # Generate a longer sequence for test target
        test_target_len = seq_len_train 
        num_cycles_test = (test_target_len + X_test_seed.shape[1] + cycle_len -1) // cycle_len
        test_full_sequence = base_pattern.repeat(num_cycles_test)
        
        # Y_test_true_full starts after X_test_seed ends
        y_test_start_index = X_test_seed.shape[1]
        Y_test_true_full = test_full_sequence[y_test_start_index : y_test_start_index + test_target_len].unsqueeze(0)

        # Ensure Y_test_true_full is not empty if test_target_len > 0
        if Y_test_true_full.shape[1] == 0 and test_target_len > 0:
            print(f"Warning: Y_test_true_full is empty for quantum_learning. Re-generating with a default.")
            # Fallback if slicing results in empty, e.g. if test_target_len is too small or pattern is short
            Y_test_true_full = base_pattern.repeat(1)[:test_target_len].unsqueeze(0)
            if Y_test_true_full.shape[1] == 0 and test_target_len > 0: # Still empty
                 Y_test_true_full = torch.tensor([[0]*test_target_len], dtype=torch.long)


        # Ensure X_test_seed is not empty
        if X_test_seed.shape[1] == 0:
            X_test_seed = torch.tensor([[0]], dtype=torch.long) # Default seed if empty


        return X_train, Y_train, X_test_seed, Y_test_true_full

    def evaluate(self, y_true, y_pred, verbose=True):
        """
        評估模型預測結果 (Adapted from classical_learning.py)
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
            # This case is handled more extensively in run_all_benchmarks.py
            # but as a safeguard within the task evaluation:
            print(f"    Task Evaluate Warning: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}. Errors might be inaccurate.")
            min_len = min(len(y_true), len(y_pred))
            if min_len == 0:
                 if verbose:
                      print(f"    Mean Absolute Error (MAE): inf")
                      print(f"    Mean Squared Error (MSE): inf")
                 return np.inf, np.inf
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        if y_true.size == 0: # Handles empty arrays after potential truncation
            if verbose:
                print(f"    Mean Absolute Error (MAE): 0.000000 (empty inputs)")
                print(f"    Mean Squared Error (MSE): 0.000000 (empty inputs)")
            return 0.0, 0.0
            
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        if verbose:
            # Using the same print format as run_all_benchmarks.py for consistency when called directly
            print(f"    Mean Absolute Error (MAE): {mae:.6f}") 
            print(f"    Mean Squared Error (MSE): {mse:.6f}")
        return mae, mse

def run_simple_learning_test():
    # 1. Define Hyper-small Model Configuration
    vocab_size = 3
    seq_len_train = 9 # e.g., 0,1,2,0,1,2,0,1,2
    n_embd_test = 8   # Keep it small, divisible by n_head
    n_head_test = 2
    n_layer_test = 1
    n_qubits_test = 2 # Min qubits, ensure classical_input_projection matches
    q_depth_test = 1
    block_size_test = seq_len_train + 5 # Should be >= any sequence length used

    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test,
        n_layer=n_layer_test,
        vocab_size=vocab_size,
        block_size=block_size_test, 
        n_intermediate=n_embd_test * 2, # Classical FFN size, less direct impact here
        layer_norm_epsilon=1e-5,
        n_qubits=n_qubits_test,
        q_depth=q_depth_test
    )
    print(f"Quantum Model Config for Learning Test: {config}\n")

    # Instantiate model
    try:
        model = QuantumRWKVModel(config)
    except Exception as e:
        print(f"Error instantiating QuantumRWKVModel: {e}")
        print("Ensure PennyLane is installed. Try: pip install pennylane")
        raise
    print("QuantumRWKVModel instantiated successfully for learning test.\n")

    # 2. Prepare Training Data (Simple repeating sequence: 0, 1, 2, 0, 1, 2, ...)
    # Input:  0, 1, 2, 0, 1, 2, 0, 1
    # Target: 1, 2, 0, 1, 2, 0, 1, 2 
    data_sequence = torch.tensor([0, 1, 2] * (seq_len_train // 3 + 1), dtype=torch.long)
    input_sequence = data_sequence[:-1].unsqueeze(0) # (B=1, T_train-1)
    target_sequence = data_sequence[1:].unsqueeze(0)  # (B=1, T_train-1)
    current_train_len = input_sequence.shape[1]

    print(f"Input sequence shape: {input_sequence.shape}, Data: {input_sequence}")
    print(f"Target sequence shape: {target_sequence.shape}, Data: {target_sequence}\n")

    # 3. Training Loop
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 200 # Increase if needed for convergence on this simple task
    print_every = 20

    model.train() # Set model to training mode
    print("Starting training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # For RWKV, states are managed internally or passed. 
        # For this simple full sequence training, we can reset states (or pass None for init)
        # However, the current model.forward expects states or initializes them.
        # Let's pass None to trigger internal initialization for each batch (epoch here).
        initial_states = None 
        
        logits, updated_states = model(input_sequence, states=initial_states)
        
        # Logits: (B, T, VocabSize), Target: (B, T)
        # Reshape for CrossEntropyLoss: Logits (B*T, VocabSize), Target (B*T)
        loss = criterion(logits.reshape(-1, vocab_size), target_sequence.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training finished.\n")

    # 4. Prediction/Generation after Training
    model.eval() # Set model to evaluation mode
    print("Starting generation...")
    
    seed_token = torch.tensor([[0]], dtype=torch.long) # Start with token 0, Batch size 1
    generated_sequence = [seed_token.item()]
    num_tokens_to_generate = seq_len_train * 2 # Generate a bit longer sequence
    
    current_input = seed_token
    # Initialize generation states (similar to training initial states)
    # For QuantumRWKVModel, states is a list of tuples per layer
    # Each tuple: ( (aa, bb, pp), prev_x_cm_state )
    # We need to get the batch size (B=1) and device from the model/seed_token
    B_gen = current_input.size(0)
    device_gen = current_input.device
    dtype_gen = model.wte.weight.dtype # Get dtype from model parameters

    generation_states = []
    for _ in range(config.n_layer):
        initial_wkv_aa = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=dtype_gen)
        initial_wkv_bb = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=dtype_gen)
        initial_wkv_pp = torch.full((B_gen, config.n_embd), -1e38, device=device_gen, dtype=dtype_gen)
        wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
        cm_state = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=dtype_gen)
        generation_states.append((wkv_state, cm_state))

    with torch.no_grad():
        for _ in range(num_tokens_to_generate -1):
            logits_gen, generation_states = model(current_input, states=generation_states)
            # Logits for the last token: logits_gen (B, T=1, VocabSize) -> (B, VocabSize)
            next_token_logits = logits_gen[:, -1, :]
            predicted_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_sequence.append(predicted_token.item())
            current_input = predicted_token # Next input is the predicted token
            
            if len(generated_sequence) >= num_tokens_to_generate:
                break

    print(f"Original pattern (first {seq_len_train} tokens): {data_sequence[:seq_len_train].tolist()}")
    print(f"Generated sequence: {generated_sequence}")

    # Check if it learned the pattern (simple check)
    expected_pattern_part = data_sequence[:len(generated_sequence)].tolist()
    if generated_sequence == expected_pattern_part:
        print("Model seems to have learned the simple periodic pattern!")
    else:
        print("Model did not perfectly reproduce the simple periodic pattern.")
        print(f"Expected : {expected_pattern_part}")
        print(f"Generated: {generated_sequence}")

def run_quantum_waveform_learning_test():
    print("\n=== Running Quantum Waveform Learning Test ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Define Model Configuration for Waveform Prediction
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
        block_size=seq_len_train + 10, # Ensure block_size is appropriate
        n_intermediate=n_embd_test * 2,
        layer_norm_epsilon=1e-5,
        input_dim=input_dim_test,
        output_dim=output_dim_test,
        n_qubits=n_qubits_test,
        q_depth=q_depth_test
        # Ensure vocab_size is NOT passed
    )
    print(f"Quantum Model Config for Waveform Learning Test: {config}\n")
    
    try:
        model = QuantumRWKVModel(config)
    except Exception as e:
        print(f"Error instantiating QuantumRWKVModel for waveform: {e}")
        print("Ensure quantum_rwkv.py's ModelConfig and QuantumRWKVModel are adapted for waveforms.")
        raise
    print("QuantumRWKVModel for waveform instantiated successfully.\n")

    # 2. Prepare Waveform Data (Sine wave)
    total_points = 500
    time_steps = np.linspace(0, 50, total_points)
    waveform = np.sin(time_steps).astype(np.float32)

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

    # 3. Training Loop (Sliding Window)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() 
    num_epochs = 1000 # Can be adjusted
    print_every = 50 # Will also be plot_every
    num_total_train_points = X_train.shape[1]

    all_epoch_losses = [] # To store loss for each epoch

    model.train()
    print("Starting training for waveform prediction (Quantum - Sliding Windows)...")
    training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_windows_processed = 0
        for i in range(num_total_train_points - seq_len_train + 1):
            optimizer.zero_grad()
            input_window = X_train[:, i : i + seq_len_train, :] 
            target_window = Y_train[:, i : i + seq_len_train, :] 
            if input_window.shape[1] != seq_len_train: # Safety check
                continue
            initial_states = None # Reset or manage states appropriately for RWKV
            predictions, _ = model(input_window, states=initial_states)
            loss = criterion(predictions, target_window)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_windows_processed += 1

        if num_windows_processed > 0:
            average_epoch_loss = epoch_loss / num_windows_processed
            all_epoch_losses.append(average_epoch_loss) # Store average loss
            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_epoch_loss:.6f}")
                
                # --- Plotting at print_every interval ---
                model.eval() # Switch to evaluation mode for plotting
                print(f"Generating plot for epoch {epoch+1}...")
                
                current_plot_generated_points = []
                plot_current_input_sequence = X_test_seed.clone() # Use the original test seed
                plot_num_points_to_generate = Y_test_true_full.shape[1]

                # Re-initialize generation states for this plotting instance
                plot_B_gen = plot_current_input_sequence.size(0)
                plot_device_gen = plot_current_input_sequence.device
                plot_param_dtype = next(model.parameters()).dtype
                plot_generation_states = []
                for _ in range(config.n_layer):
                    p_initial_wkv_aa = torch.zeros(plot_B_gen, config.n_embd, device=plot_device_gen, dtype=plot_param_dtype)
                    p_initial_wkv_bb = torch.zeros(plot_B_gen, config.n_embd, device=plot_device_gen, dtype=plot_param_dtype)
                    p_initial_wkv_pp = torch.full((plot_B_gen, config.n_embd), -1e38, device=plot_device_gen, dtype=plot_param_dtype)
                    p_wkv_state = (p_initial_wkv_aa, p_initial_wkv_bb, p_initial_wkv_pp)
                    p_cm_state = torch.zeros(plot_B_gen, config.n_embd, device=plot_device_gen, dtype=plot_param_dtype)
                    plot_generation_states.append((p_wkv_state, p_cm_state))

                with torch.no_grad():
                    for _ in range(plot_num_points_to_generate):
                        plot_pred_out, plot_generation_states = model(plot_current_input_sequence, states=plot_generation_states)
                        plot_next_pred_point = plot_pred_out[:, -1, :].clone()
                        current_plot_generated_points.append(plot_next_pred_point.squeeze().item())
                        plot_current_input_sequence = torch.cat((plot_current_input_sequence[:, 1:, :], plot_next_pred_point.unsqueeze(1)), dim=1)
                
                plot_generated_waveform_tensor = torch.tensor(current_plot_generated_points, dtype=torch.float32)
                # True waveform for plotting remains the same
                plot_true_waveform_part_for_eval = Y_test_true_full.squeeze().cpu().numpy()
                plot_generated_waveform_for_eval = plot_generated_waveform_tensor.cpu().numpy()

                # Ensure lengths match for plotting, if generation is shorter/longer for some reason
                min_len_plot = min(len(plot_generated_waveform_for_eval), len(plot_true_waveform_part_for_eval))
                plot_generated_waveform_for_eval = plot_generated_waveform_for_eval[:min_len_plot]
                plot_true_waveform_part_for_eval_adjusted = plot_true_waveform_part_for_eval[:min_len_plot]

                plt.figure(figsize=(14, 7))
                plot_time_steps_plot_true = np.arange(len(plot_true_waveform_part_for_eval_adjusted))
                plot_time_steps_plot_gen = np.arange(len(plot_generated_waveform_for_eval))
                plt.plot(plot_time_steps_plot_true, plot_true_waveform_part_for_eval_adjusted, label='Ground Truth Waveform', color='blue', linestyle='-')
                plt.plot(plot_time_steps_plot_gen, plot_generated_waveform_for_eval, label=f'Predicted Waveform (Epoch {epoch+1})', color='red', linestyle='--')
                plt.title(f'Ground Truth vs. Predicted Waveform (Quantum RWKV - Epoch {epoch+1})')
                plt.xlabel('Time Step (in test segment)')
                plt.ylabel('Waveform Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Ensure results_dir exists (it's created later, but good to have it for epoch plots too)
                results_dir_epoch_plot = "results"
                os.makedirs(results_dir_epoch_plot, exist_ok=True)
                plot_filename_epoch = os.path.join(results_dir_epoch_plot, f"waveform_comparison_quantum_epoch_{epoch+1}.png")
                try:
                    plt.savefig(plot_filename_epoch)
                    print(f"Plot saved as {plot_filename_epoch}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving epoch plot: {e}")
                model.train() # Switch back to training mode
                # --- End Plotting --- 

        elif (epoch + 1) % print_every == 0: # Handle case where no windows were processed
             print(f"Epoch [{epoch+1}/{num_epochs}], No windows processed.")
             all_epoch_losses.append(float('nan')) # Store NaN if no windows

        if num_windows_processed > 0:
            average_epoch_loss = epoch_loss / num_windows_processed
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
                results_dir = "results_waveform_quantum_simple"
                os.makedirs(results_dir, exist_ok=True)
                plot_filename = os.path.join(results_dir, f"waveform_prediction_comparison_quantum_rwkv_{training_start_time}_epoch{epoch+1}.png")
                try:
                    plt.savefig(plot_filename)
                    print(f"Plot saved as {plot_filename}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving or showing plot: {e}")
                model.train()
    print("Training finished.\n")

    # 4. Prediction/Generation and Evaluation (This is the FINAL evaluation)
    model.eval()
    print("Starting generation for waveform prediction (Quantum)...")

    generated_waveform_points = []
    current_input_sequence = X_test_seed.clone()
    num_points_to_generate = Y_test_true_full.shape[1]

    B_gen = current_input_sequence.size(0)
    device_gen = current_input_sequence.device
    param_dtype = next(model.parameters()).dtype # Match model's dtype
    generation_states = []
    for _ in range(config.n_layer):
        initial_wkv_aa = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=param_dtype)
        initial_wkv_bb = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=param_dtype)
        initial_wkv_pp = torch.full((B_gen, config.n_embd), -1e38, device=device_gen, dtype=param_dtype) # Log space
        wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
        cm_state = torch.zeros(B_gen, config.n_embd, device=device_gen, dtype=param_dtype) # For channel mixing
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

    mae, mse = model.evaluate(true_waveform_part_for_eval, generated_waveform_for_eval)

    print(f"Generated Waveform (Quantum, first 20 points): {generated_waveform_for_eval[:20].tolist()}")
    print(f"True Waveform (Quantum, first 20 points):      {true_waveform_part_for_eval[:20].tolist()}")
    print(f"Mean Absolute Error (MAE, Quantum): {mae:.6f}")
    print(f"Mean Squared Error (MSE, Quantum):  {mse:.6f}\n")

    if mse < 0.1: 
        print("Quantum model shows some basic learning on waveform prediction.")
    else:
        print("Quantum model did not significantly learn the waveform pattern (MSE > 0.1).")
        
    results_dir = "results_waveform_quantum"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(14, 7))
    plot_time_steps_true = np.arange(len(true_waveform_part_for_eval))
    plot_time_steps_gen = np.arange(len(generated_waveform_for_eval))
    plt.plot(plot_time_steps_true, true_waveform_part_for_eval, label='Ground Truth Waveform', color='blue', linestyle='-')
    plt.plot(plot_time_steps_gen, generated_waveform_for_eval, label='Predicted Waveform (Quantum)', color='red', linestyle='--')
    plt.title('Ground Truth vs. Predicted Waveform (Quantum RWKV)')
    plt.xlabel('Time Step (in test segment)')
    plt.ylabel('Waveform Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = os.path.join(results_dir, f"waveform_prediction_comparison_quantum_rwkv_{training_start_time}_final.png")
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.close() 
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    # --- Enhanced Metrics Saving to CSV ---
    csv_filename = os.path.join(results_dir, "model_performance.csv")
    
    # New header
    header = [
        'Timestamp', 'Experiment_ID', 'Model_Type', 'Task', 
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth', 
        'learning_rate', 'num_epochs_run', 'seq_len_train',
        'Config_Block_Size', 'Config_n_intermediate', 
        'MAE', 'MSE'
    ]
    
    # Prepare data for the new row
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = f"q_wave_{timestamp}"
    learning_rate = optimizer.param_groups[0]['lr'] # Get LR from optimizer

    data_row = [
        timestamp, experiment_id, 'Quantum', 'Waveform',
        config.n_layer, config.n_embd, config.n_head, config.n_qubits, config.q_depth,
        f'{learning_rate:.1e}', num_epochs, seq_len_train, # Using num_epochs as num_epochs_run
        config.block_size, config.n_intermediate,
        f'{mae:.6f}', f'{mse:.6f}'
    ]
    
    file_exists = os.path.isfile(csv_filename)
    is_empty = os.path.getsize(csv_filename) == 0 if file_exists else True
    
    try:
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile: 
            writer = csv.writer(csvfile)
            if not file_exists or is_empty: # Write header if file is new or empty
                writer.writerow(header)
            writer.writerow(data_row) 
        print(f"Detailed metrics saved to {csv_filename}")
    except Exception as e:
        print(f"Error writing detailed metrics to CSV {csv_filename}: {e}")
        traceback.print_exc() 
        
    # --- Save Epoch Losses to CSV ---
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
        traceback.print_exc()

    print("\n=== Finished Quantum Waveform Learning Test ===\n")

if __name__ == '__main__':
    run_simple_learning_test()
    run_quantum_waveform_learning_test() 