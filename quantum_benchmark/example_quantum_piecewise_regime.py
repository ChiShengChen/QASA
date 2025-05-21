import torch
from quantum_rwkv import get_task
from quantum_rwkv import ModelConfig, QuantumRWKVModel

# 1. 取得 quantum 任務
task = get_task('quantum_piecewise_regime')

# 2. 產生資料
X_train, Y_train, X_test_seed, Y_test_true_full = task.generate_data()

# 3. 建立 quantum 模型
config = ModelConfig(
    n_embd=16,
    n_head=2,
    n_layer=1,
    block_size=30,
    n_intermediate=32,
    layer_norm_epsilon=1e-5,
    input_dim=1,
    output_dim=1,
    n_qubits=4,
    q_depth=1
)
model = QuantumRWKVModel(config)

# 4. 自動化訓練流程
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
model.train()
seq_len_train = 20
for i in range(X_train.shape[1] - seq_len_train + 1):
    optimizer.zero_grad()
    input_window = X_train[:, i:i+seq_len_train, :]
    target_window = Y_train[:, i:i+seq_len_train, :]
    pred, _ = model(input_window)
    loss = criterion(pred, target_window)
    loss.backward()
    optimizer.step()

# 5. 預測
model.eval()
with torch.no_grad():
    current_input = X_test_seed.clone()
    num_points = Y_test_true_full.shape[1]
    generated = []
    for _ in range(num_points):
        pred, _ = model(current_input)
        next_point = pred[:, -1, :].clone()
        generated.append(next_point.squeeze().item())
        current_input = torch.cat((current_input[:, 1:, :], next_point.unsqueeze(1)), dim=1)
    generated = torch.tensor(generated, dtype=torch.float32)

# 6. 評估
y_true = Y_test_true_full.squeeze().cpu().numpy()
y_pred = generated.cpu().numpy()
task.evaluate(y_true, y_pred) 