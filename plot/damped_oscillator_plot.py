import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 設置字體為系統默認，避免警告
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size": 14,
    "figure.figsize": (10, 5.5),
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.frameon": False,
})

# 高DPI設置
plt.figure(dpi=300)

# 參數設置
A = 1.0        # 振幅
gamma = 0.1    # 阻尼係數
omega = 2.0    # 角頻率
phi = 0        # 初始相位
    
# 生成時間序列
t = np.linspace(0, 40, 1000)
    
# 計算阻尼諧振子位置
x = A * np.exp(-gamma * t) * np.cos(omega * t + phi)

# 創建圖形
fig, ax = plt.subplots(figsize=(10, 5))

# 繪製阻尼振盪器曲線，增加線條寬度和使用深藍色
ax.plot(t, x, linewidth=3.8, color='#1f77b4')

# 設置坐標軸標籤
ax.set_xlabel('Time (t)', fontsize=16, labelpad=10)
ax.set_ylabel('Amplitude x(t)', fontsize=16, labelpad=10)

# 移除標題，讓論文圖表更簡潔
# ax.set_title('Damped Harmonic Oscillator', fontsize=18, pad=15)

# 添加公式說明，使用數學方式表示
equation = r"$x(t) = Ae^{-\gamma t}\cos(\omega t + \phi)$"
# 調整位置到左上角
ax.text(0.65, 0.9, equation, transform=ax.transAxes, 
        fontsize=18, bbox=dict(facecolor='white', alpha=0.7, 
                               boxstyle='round,pad=0.5', edgecolor='none'))

# 移除上方和右方的邊框，保持簡潔風格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 加粗左邊和底部的坐標軸線
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 設置坐標軸刻度的寬度
ax.tick_params(width=1.5, length=6)

# 設置坐標軸範圍
ax.set_xlim(0, 40)
ax.set_ylim(-1.1, 1.1)

# 添加灰色參考線但較淡
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2, linewidth=1)

# 調整佈局並保存
plt.tight_layout()
plt.savefig('damped_oscillator.png', dpi=300, bbox_inches='tight')
plt.savefig('damped_oscillator.pdf', bbox_inches='tight')  # 同時保存PDF格式，適合論文

print("已更新並保存阻尼振盪器圖像") 