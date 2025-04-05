import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from CSV files
quantum_data = pd.read_csv('EEGPT/downstream/csv_logs/quantum_new_v4/training_log_20250330_230032.csv')
classical_data = pd.read_csv('EEGPT/downstream/csv_logs/classical_v4/training_log_20250330_224623.csv')

# Limit data to first 45 epochs
quantum_data = quantum_data.iloc[:45]
classical_data = classical_data.iloc[:45]

# Create a figure
plt.figure(figsize=(14, 10))

# Plot training losses
plt.plot(quantum_data['epoch'], quantum_data['train_loss'], 'b-', linewidth=3, 
         label='Quantum - Training Loss', alpha=0.9)
plt.plot(classical_data['epoch'], classical_data['train_loss'], 'r-', linewidth=3, 
         label='Classical - Training Loss', alpha=0.9)

# Plot validation losses
plt.plot(quantum_data['epoch'], quantum_data['val_loss'], 'b--', linewidth=2, 
         label='Quantum - Validation Loss', alpha=0.7)
plt.plot(classical_data['epoch'], classical_data['val_loss'], 'r--', linewidth=2, 
         label='Classical - Validation Loss', alpha=0.7)

# Add labels and title
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Training and Validation Loss Comparison: Quantum vs Classical Models', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14, loc='upper right')

# Set axis limits
plt.xlim(0, 45)
plt.ylim(0, 1.2)

# Add a vertical line at epoch 18 where significant improvement begins
plt.axvline(x=18, color='gray', linestyle='--', alpha=0.5)
plt.text(18.5, 1.1, 'Significant improvement begins', fontsize=12, rotation=0)

# Highlight minimum validation loss points
min_quantum_val = quantum_data['val_loss'].min()
min_quantum_val_epoch = quantum_data.loc[quantum_data['val_loss'].idxmin(), 'epoch']
min_classical_val = classical_data['val_loss'].min()
min_classical_val_epoch = classical_data.loc[classical_data['val_loss'].idxmin(), 'epoch']

plt.scatter(min_quantum_val_epoch, min_quantum_val, color='blue', s=150, 
            edgecolor='black', zorder=5, label='_nolegend_')
plt.scatter(min_classical_val_epoch, min_classical_val, color='red', s=150, 
            edgecolor='black', zorder=5, label='_nolegend_')

plt.annotate(f'Min Quantum Val Loss: {min_quantum_val:.4f}', 
             xy=(min_quantum_val_epoch, min_quantum_val),
             xytext=(min_quantum_val_epoch-7, min_quantum_val-0.15),
             arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.7),
             fontsize=12, fontweight='bold')

plt.annotate(f'Min Classical Val Loss: {min_classical_val:.4f}', 
             xy=(min_classical_val_epoch, min_classical_val),
             xytext=(min_classical_val_epoch-7, min_classical_val-0.1),
             arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7),
             fontsize=12, fontweight='bold')

# Add comparative metrics table
model_metrics = {
    'Model': ['Quantum', 'Classical'],
    'Min Train Loss': [quantum_data['train_loss'].min(), classical_data['train_loss'].min()],
    'Min Val Loss': [min_quantum_val, min_classical_val],
    'Improvement Ratio': [quantum_data['train_loss'].iloc[0]/quantum_data['train_loss'].min(), 
                           classical_data['train_loss'].iloc[0]/classical_data['train_loss'].min()]
}

cell_text = [
    [f"{model_metrics['Model'][0]}", f"{model_metrics['Min Train Loss'][0]:.4f}", 
     f"{model_metrics['Min Val Loss'][0]:.4f}", f"{model_metrics['Improvement Ratio'][0]:.2f}"],
    [f"{model_metrics['Model'][1]}", f"{model_metrics['Min Train Loss'][1]:.4f}", 
     f"{model_metrics['Min Val Loss'][1]:.4f}", f"{model_metrics['Improvement Ratio'][1]:.2f}"]
]

plt.table(cellText=cell_text,
          colLabels=['Model', 'Min Train Loss', 'Min Val Loss', 'Improvement Ratio'],
          loc='bottom',
          bbox=[0.15, -0.25, 0.7, 0.15],
          cellLoc='center')

# Add text with key observations 
plt.figtext(0.5, 0.02, 
           'Key Observations:\n'
           '1. Both models show similar training patterns but different convergence rates\n'
           '2. The quantum model achieves lower final training and validation loss\n'
           '3. The classical model shows more validation loss fluctuation after epoch 30',
           ha='center', fontsize=14, bbox=dict(facecolor='lightyellow', alpha=0.5))

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)  # Make room for the table and text
plt.savefig('EEGPT/downstream/combined_loss_comparison.png', dpi=300)
plt.show()

print("Combined loss comparison plot created and saved as 'combined_loss_comparison.png'") 