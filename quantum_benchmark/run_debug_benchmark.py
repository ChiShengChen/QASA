import sys
import os

# Add the QASA-main directory (project root) to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the Quantum_rwkv directory to sys.path
quantum_rwkv_dir = os.path.join(os.path.expanduser("~"), "Quantum_rwkv")
if os.path.exists(quantum_rwkv_dir) and quantum_rwkv_dir not in sys.path:
    sys.path.insert(0, quantum_rwkv_dir)
elif not os.path.exists(quantum_rwkv_dir):
    project_root_parent = os.path.dirname(project_root)
    quantum_rwkv_dir_alt = os.path.join(project_root_parent, "Quantum_rwkv")
    if os.path.exists(quantum_rwkv_dir_alt) and quantum_rwkv_dir_alt not in sys.path:
        sys.path.insert(0, quantum_rwkv_dir_alt)
        print(f"Note: Added {quantum_rwkv_dir_alt} to sys.path for rwkv module.", file=sys.stderr)
    else:
        print(f"Warning: Directory {quantum_rwkv_dir} (and alternative) not found. The 'rwkv' module might not be importable.", file=sys.stderr)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import datetime # Keep datetime import
import traceback # Keep traceback import

# Assuming task_registry and get_task are correctly imported due to sys.path modification
from QASA.quantum_benchmark.tasks import task_registry, get_task 
# Assuming ModelConfig and RWKVModel are correctly imported (rwkv.py in PYTHONPATH or sys.path)
from rwkv import ModelConfig, RWKVModel 

# --- Configuration (Copied from run_all_benchmarks.py) ---
RESULTS_FILE = "../all_benchmark_results_debug.csv" # Use a different CSV for debug results
N_EMBD = 16
N_HEAD = 2
N_LAYER = 1
BLOCK_SIZE_WAVEFORM = 30
BLOCK_SIZE_TOKEN = 20 
VOCAB_SIZE_DEFAULT = 50 
N_INTERMEDIATE_FACTOR = 2
LAYER_NORM_EPSILON = 1e-5
LEARNING_RATE = 1e-3
NUM_EPOCHS_WAVEFORM = 10 
NUM_EPOCHS_TOKEN = 50    
SEQ_LEN_TRAIN_WAVEFORM = 20

# --- run_single_task function (Copied from run_all_benchmarks.py) ---
def run_single_task(task_name, device):
    print(f"\n--- Running task: {task_name} ---")
    task = get_task(task_name)

    is_learning_task = "learning" in task_name.lower()
    
    if is_learning_task:
        vocab_size_task = getattr(task, 'vocab_size', VOCAB_SIZE_DEFAULT) 
        X_train, Y_train, X_test_seed, Y_test_true_full = task.generate_data(vocab_size=vocab_size_task)
    else:
        X_train, Y_train, X_test_seed, Y_test_true_full = task.generate_data()

    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test_seed, Y_test_true_full = X_test_seed.to(device), Y_test_true_full.to(device)

    if is_learning_task:
        actual_vocab_size = X_train.max().item() + 1 if X_train.numel() > 0 else VOCAB_SIZE_DEFAULT
        if Y_train.numel() > 0:
             actual_vocab_size = max(actual_vocab_size, Y_train.max().item() + 1)
        print(f"Inferred vocab_size for {task_name}: {actual_vocab_size}")
        config = ModelConfig(
            n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE_TOKEN,
            vocab_size=actual_vocab_size, n_intermediate=N_EMBD * N_INTERMEDIATE_FACTOR,
            layer_norm_epsilon=LAYER_NORM_EPSILON)
        criterion = nn.CrossEntropyLoss()
    else:
        input_dim = X_train.shape[-1] if X_train.ndim > 2 and X_train.shape[-1] > 0 else 1
        output_dim = Y_train.shape[-1] if Y_train.ndim > 2 and Y_train.shape[-1] > 0 else 1
        print(f"Inferred input_dim: {input_dim}, output_dim: {output_dim} for {task_name}")
        config = ModelConfig(
            n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE_WAVEFORM,
            input_dim=input_dim, output_dim=output_dim, n_intermediate=N_EMBD * N_INTERMEDIATE_FACTOR,
            layer_norm_epsilon=LAYER_NORM_EPSILON)
        criterion = nn.MSELoss()

    model = RWKVModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    num_epochs = NUM_EPOCHS_TOKEN if is_learning_task else NUM_EPOCHS_WAVEFORM
    
    for epoch in range(num_epochs):
        if not is_learning_task and X_train.shape[1] > SEQ_LEN_TRAIN_WAVEFORM :
            epoch_loss = 0; num_batches = 0
            for i in range(X_train.shape[1] - SEQ_LEN_TRAIN_WAVEFORM + 1):
                optimizer.zero_grad()
                input_window = X_train[:, i:i+SEQ_LEN_TRAIN_WAVEFORM]; target_window = Y_train[:, i:i+SEQ_LEN_TRAIN_WAVEFORM]
                pred, _ = model(input_window)
                loss = criterion(pred, target_window); loss.backward(); optimizer.step()
                epoch_loss += loss.item(); num_batches +=1
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        else: 
            optimizer.zero_grad(); pred, _ = model(X_train)
            loss = criterion(pred.reshape(-1, pred.shape[-1]) if is_learning_task else pred, Y_train.reshape(-1) if is_learning_task else Y_train)
            loss.backward(); optimizer.step(); avg_epoch_loss = loss.item()
        if (epoch + 1) % 10 == 0 or epoch == num_epochs -1:
            print(f"Task: {task_name}, Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    model.eval(); generated_outputs = []
    with torch.no_grad():
        current_input_seed = X_test_seed.clone()
        num_points_to_predict = Y_test_true_full.shape[1]
        if is_learning_task:
            current_tokens = current_input_seed
            if current_tokens.shape[1] == 0 and num_points_to_predict > 0 :
                 current_tokens = torch.zeros((X_test_seed.shape[0], 1), dtype=torch.long, device=device)
            for _ in range(num_points_to_predict):
                if current_tokens.shape[1] == 0: print(f"Warning: current_tokens empty for {task_name}."); break
                pred_logits, _ = model(current_tokens) 
                next_token = torch.argmax(pred_logits[:, -1, :], dim=-1, keepdim=True)
                generated_outputs.append(next_token.squeeze().item())
                current_tokens = next_token 
                if len(generated_outputs) >= num_points_to_predict: break
            y_pred_final = torch.tensor(generated_outputs, dtype=Y_test_true_full.dtype, device=device)
        else: 
            for _ in range(num_points_to_predict):
                if current_input_seed.shape[1] == 0: print(f"Warning: current_input_seed empty for {task_name}."); break
                pred_continuous, _ = model(current_input_seed)
                next_point = pred_continuous[:, -1, :].clone()
                generated_outputs.append(next_point.squeeze().cpu().numpy())
                current_input_seed = torch.cat((current_input_seed[:, 1:, :], next_point.unsqueeze(1)), dim=1)
                if len(generated_outputs) >= num_points_to_predict: break
            y_pred_final = torch.tensor(np.array(generated_outputs), dtype=Y_test_true_full.dtype, device=device)
            if y_pred_final.ndim == 1 and Y_test_true_full.ndim == 2 and Y_test_true_full.shape[0] == 1: y_pred_final = y_pred_final.unsqueeze(0)
            elif y_pred_final.ndim > 1 and y_pred_final.shape[0]!=Y_test_true_full.shape[0] and Y_test_true_full.shape[0]==1: y_pred_final = y_pred_final.unsqueeze(0)

    y_true_np = Y_test_true_full.squeeze(0).cpu().numpy()
    y_pred_np = y_pred_final.squeeze(0).cpu().numpy()
    if y_true_np.ndim > 1 and y_true_np.shape[0] == 1 : y_true_np = y_true_np.squeeze(0)
    if y_pred_np.ndim > 1 and y_pred_np.shape[0] == 1 : y_pred_np = y_pred_np.squeeze(0)
    if y_true_np.ndim > 1 and y_true_np.shape[-1] == 1 : y_true_np = y_true_np.squeeze(-1)
    if y_pred_np.ndim > 1 and y_pred_np.shape[-1] == 1 : y_pred_np = y_pred_np.squeeze(-1)

    mae, mse = np.inf, np.inf
    if y_pred_np.size == 0 and y_true_np.size > 0: print(f"Warning: No predictions for {task_name}.")
    elif y_true_np.size == 0: print(f"Warning: No true data for {task_name}."); mae, mse = 0,0
    else:
        if y_pred_np.shape[0] > y_true_np.shape[0]: y_pred_np = y_pred_np[:y_true_np.shape[0]]
        elif y_pred_np.shape[0] < y_true_np.shape[0]: y_true_np = y_true_np[:y_pred_np.shape[0]]
        if y_pred_np.size > 0 and y_true_np.size > 0 and y_pred_np.shape == y_true_np.shape: mae, mse = task.evaluate(y_true_np, y_pred_np, verbose=True)
        elif y_pred_np.size > 0 and y_true_np.size > 0 : print(f"Warning: Shape mismatch. True: {y_true_np.shape}, Pred: {y_pred_np.shape}")
        else: print(f"Warning: Zero size array for eval. True: {y_true_np.shape}, Pred: {y_pred_np.shape}")

    print(f"Finished task: {task_name}, MAE: {mae:.6f}, MSE: {mse:.6f}")
    return task_name, mae, mse

# --- main function (Modified for debug) ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # DEBUG: Only run these specific tasks
    debug_task_names = ['quantum_damped_oscillation', 'quantum_learning']
    all_task_names = [name for name in debug_task_names if name in task_registry]
    if len(all_task_names) != len(debug_task_names):
        missing = [name for name in debug_task_names if name not in task_registry]
        print(f"Warning: Some debug tasks not found in registry: {missing}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    fieldnames = ['task', 'classical_mae', 'classical_mse', 'quantum_mae', 'quantum_mse']
    existing_results = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, mode='r', newline='') as f_in:
                reader = csv.DictReader(f_in)
                if reader.fieldnames and all(f in reader.fieldnames for f in fieldnames):
                    for row in reader:
                        existing_results[row['task']] = row
                else: existing_results = {}
        except Exception as e:
            print(f"Error reading {RESULTS_FILE}: {e}. Will create anew."); existing_results = {}

    for task_name_reg in all_task_names: 
        if task_name_reg not in existing_results:
            existing_results[task_name_reg] = {fn: '' for fn in fieldnames}
            existing_results[task_name_reg]['task'] = task_name_reg

    for task_name in all_task_names:
        try:
            _, mae, mse = run_single_task(task_name, device)
            existing_results[task_name]['classical_mae'] = f"{mae:.6f}" if mae != np.inf else "inf"
            existing_results[task_name]['classical_mse'] = f"{mse:.6f}" if mse != np.inf else "inf"
        except Exception as e:
            print(f"Error running task {task_name}: {e}")
            traceback.print_exc()
            existing_results[task_name]['classical_mae'] = "error"
            existing_results[task_name]['classical_mse'] = "error"
        
        temp_file = RESULTS_FILE + ".tmp"
        with open(temp_file, mode='w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            # For debug script, only write the tasks that were run or intended to run
            sorted_tasks_to_write = sorted([tn for tn in debug_task_names if tn in existing_results])
            for tn_key in sorted_tasks_to_write:
                 writer.writerow(existing_results[tn_key])
        os.replace(temp_file, RESULTS_FILE)

    print("\n--- Debug tasks finished ---")
    print(f"Results saved to {os.path.abspath(RESULTS_FILE)}")

if __name__ == "__main__":
    main() 