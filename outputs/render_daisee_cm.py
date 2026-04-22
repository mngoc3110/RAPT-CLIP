import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Data from DAiSEE screenshot
class_names = ['Disengaged', 'Engaged', 'Highly\nEngaged']
cm_counts = np.array([
    [19, 141, 6],
    [28, 744, 41],
    [12, 373, 65]
])

cm_percentages = np.array([
    [11.4, 84.9, 3.6],
    [3.4, 91.5, 5.0],
    [2.7, 82.9, 14.4]
])

uar = 39.13
war = 57.94

# Normalize for colors (using the percentage array / 100)
cm_norm = cm_percentages / 100.0

# Create annotation text: count + percentage
annot = np.empty_like(cm_counts, dtype=object)
for i in range(cm_counts.shape[0]):
    for j in range(cm_counts.shape[1]):
        count = cm_counts[i, j]
        pct = cm_percentages[i, j]
        annot[i, j] = f'{pct:.1f}%\n({count})'

# Plot
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
})

fig, ax = plt.subplots(figsize=(8, 6.5))

sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, vmin=0, vmax=1.0,
            linewidths=1.0, linecolor='black', square=True,
            annot_kws={"size": 13, "weight": "bold"},
            cbar_kws={"label": "Recall (%)", "shrink": 0.8})

ax.set_title(f'Confusion Matrix - DAiSEE (Validation)\nUAR: {uar}% | WAR: {war}%',
             fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')

plt.tight_layout()

out_dir = 'outputs/DAiSEE'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
print("Saved: outputs/DAiSEE/confusion_matrix.png")
