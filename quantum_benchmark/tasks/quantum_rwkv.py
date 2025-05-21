import torch
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
# Make sure to import from the new quantum_rwkv.py file
from quantum_rwkv import ModelConfig, QuantumRWKVModel 

class Task:
    """
    時序任務範本。
    - 實作資料生成、評估等方法
    - 不含模型本身
    """

    def generate_data(self, *args, **kwargs):
        """
        產生訓練/測試資料
        """
        raise NotImplementedError

    def evaluate(self, y_true, y_pred, *args, **kwargs):
        """
        評估模型預測結果
        """
        raise NotImplementedError

def test_quantum_model_forward():
    # Test parameters
    batch_size = 2
    seq_len = 5 # Shorter sequence for potentially slower quantum simulation
    vocab_size = 50
    n_embd_test = 32 # Must be divisible by n_head_test
    n_head_test = 4  # For RWKVTimeMixing
    n_layer_test = 2 # Fewer layers for faster test
    
    # Quantum-specific parameters for ModelConfig
    n_qubits_test = 4 
    q_depth_test = 1 # Shallow depth for faster test
    
    # Create a model configuration for the quantum model
    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test, # Classical TimeMixing still needs this
        n_layer=n_layer_test,
        vocab_size=vocab_size,
        block_size=seq_len + 10, 
        n_intermediate=n_embd_test * 2, # Less relevant for QFFN, but part of config
        layer_norm_epsilon=1e-5,
        n_qubits=n_qubits_test,
        q_depth=q_depth_test
    )
    
    print(f"Quantum Model Config: {config}\n")
    
    # Instantiate the quantum model
    # This might take a bit longer due to PennyLane VQC initialization
    try:
        model = QuantumRWKVModel(config)
        model.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error instantiating QuantumRWKVModel: {e}")
        print("Ensure PennyLane is installed and configured correctly.")
        print("Try: pip install pennylane pennylane-lightning[torch] (or other backends)")
        raise
        
    print(f"QuantumRWKVModel instantiated successfully.\n")
    
    # Create dummy input token indices (Batch, SeqLen)
    dummy_idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Dummy input idx shape: {dummy_idx.shape}\n")
    
    # --- First forward pass (no initial states) ---
    print("--- First forward pass (states=None) for Quantum Model ---")
    try:
        logits1, states1 = model(dummy_idx, states=None)
    except Exception as e:
        print(f"Error during first forward pass of QuantumRWKVModel: {e}")
        raise

    print(f"Logits shape (Quantum): {logits1.shape}")
    assert logits1.shape == (batch_size, seq_len, vocab_size)
    
    print(f"Number of layers for states (Quantum): {len(states1)}")
    assert len(states1) == n_layer_test
    
    for i, layer_state in enumerate(states1):
        wkv_state, cm_state = layer_state
        aa, bb, pp = wkv_state
        print(f"  Layer {i} States (Quantum):")
        print(f"    WKV state (aa, bb, pp) shapes: {aa.shape}, {bb.shape}, {pp.shape}")
        print(f"    Channel Mix state shape: {cm_state.shape}")
        assert aa.shape == (batch_size, n_embd_test)
        assert bb.shape == (batch_size, n_embd_test)
        assert pp.shape == (batch_size, n_embd_test)
        assert cm_state.shape == (batch_size, n_embd_test)
    print("\nSuccessfully completed first forward pass and state checks (Quantum).\n")

    # --- Second forward pass (using states from the first pass) ---
    print("--- Second forward pass (using states from first pass) for Quantum Model ---")
    next_dummy_idx = torch.randint(0, vocab_size, (batch_size, 1)) # Next single token
    print(f"Next dummy input idx shape (Quantum): {next_dummy_idx.shape}\n")
    
    try:
        logits2, states2 = model(next_dummy_idx, states=states1)
    except Exception as e:
        print(f"Error during second forward pass of QuantumRWKVModel: {e}")
        raise
        
    print(f"Logits shape (Quantum, second pass): {logits2.shape}")
    assert logits2.shape == (batch_size, 1, vocab_size)
    
    print(f"Number of layers for states (Quantum, second pass): {len(states2)}")
    assert len(states2) == n_layer_test
    
    for i, layer_state in enumerate(states2):
        wkv_state, cm_state = layer_state
        aa, bb, pp = wkv_state
        print(f"  Layer {i} States (Quantum, second pass):")
        print(f"    WKV state (aa, bb, pp) shapes: {aa.shape}, {bb.shape}, {pp.shape}")
        print(f"    Channel Mix state shape: {cm_state.shape}")
        assert aa.shape == (batch_size, n_embd_test)
        assert bb.shape == (batch_size, n_embd_test)
        assert pp.shape == (batch_size, n_embd_test)
        assert cm_state.shape == (batch_size, n_embd_test)
    print("\nSuccessfully completed second forward pass with states and checks (Quantum).\n")
    
    print("All quantum model tests passed!")

def run_quantum_rwkv_prediction():
    # ... existing code ...
    training_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # ... existing code ...
    for epoch in range(num_epochs):
        # ... existing code ...
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
                plt.plot(np.arange(len(true_waveform_part_for_eval)), true_waveform_part_for_eval, label='Ground Truth RWKV', color='blue', linestyle='-')
                plt.plot(np.arange(len(generated_waveform_for_eval)), generated_waveform_for_eval, label='Predicted RWKV (Quantum)', color='red', linestyle='--')
                plt.title(f'Ground Truth vs. Predicted RWKV (Quantum) - Epoch {epoch+1}')
                plt.xlabel('Time Step (in test segment)')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_filename = os.path.join(results_dir, f"rwkv_comparison_quantum_{training_start_time}_epoch{epoch+1}.png")
                try:
                    plt.savefig(plot_filename)
                    print(f"Plot saved as {plot_filename}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving plot: {e}")
                model.train()
    print("Quantum training finished.\n")
    # ... existing code ...
    plot_filename = os.path.join(results_dir, f"rwkv_comparison_quantum_{training_start_time}_final.png")
    # ... existing code ...

if __name__ == '__main__':
    test_quantum_model_forward() 