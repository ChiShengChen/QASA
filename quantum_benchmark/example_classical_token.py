import torch
from quantum_rwkv import get_task
from rwkv import ModelConfig, RWKVModel

# 1. 取得 token 任務
task = get_task('classical_learning')

# 2. 產生資料（這裡是簡單的重複序列）
input_seq, target_seq = task.generate_data(seq_len_train=9, vocab_size=3)

# 3. 建立 token 模型
config = ModelConfig(
    n_embd=8,
    n_head=2,
    n_layer=1,
    block_size=14,  # seq_len_train + 5
    n_intermediate=16,
    layer_norm_epsilon=1e-5,
    vocab_size=3
)
model = RWKVModel(config)

# 4. 自動化訓練流程
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
model.train()
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits, _ = model(input_seq)
    loss = criterion(logits.reshape(-1, config.vocab_size), target_seq.reshape(-1))
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. 預測/生成
model.eval()
with torch.no_grad():
    seed_token = torch.tensor([[0]], dtype=torch.long)
    generated = [seed_token.item()]
    current_input = seed_token
    for _ in range(18):  # 產生 18 個 token
        logits, _ = model(current_input)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated.append(next_token.item())
        current_input = next_token
    print(f"Generated sequence: {generated}")

# 6. 評估
expected = input_seq[0].tolist() + [target_seq[0, -1].item()]
task.evaluate(torch.tensor(expected), torch.tensor(generated[:len(expected)])) 