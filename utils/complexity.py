import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'

FONT_TITLE = 9.5
FONT_LABEL = 8.5
FONT_TICK = 7.5
FONT_LEGEND = 7.5

fig = plt.figure(figsize=(7.5, 3.2), dpi=300)

ax1 = fig.add_subplot(1, 2, 1)

evaluations = 5000  # 50 particles * 100 iterations
drl_total_hours = (1.34 * evaluations) / 3600
mpc_total_hours = (11.22 * evaluations) / 3600
dp_1d_hours = (5.92 * evaluations) / 3600
dp_2d_hours = (296.2 * evaluations) / 3600
dp_3d_hours = (29600.0 * evaluations) / 3600

categories = ['Proposed\nDRL', 'MPC\n($H=24$)', 'DP\n(1-D)', 'DP\n(2-D)', 'DP\n(3-D)']
hours = [drl_total_hours, mpc_total_hours, dp_1d_hours, dp_2d_hours, dp_3d_hours]
colors = ['#2ca02c', '#d62728', '#1f77b4', '#aec7e8', '#4682b4']

bars = ax1.bar(categories, hours, color=colors, width=0.55, edgecolor='black', linewidth=0.7)

ax1.set_yscale('log')
ax1.set_ylim(0.5, 2e5)
ax1.set_ylabel('Total Co-Design Optimization Time (Hours)', fontsize=FONT_LABEL, fontweight='bold')
ax1.set_title('(a) Full MOPSO Co-Design Time (5000 Evals)', fontsize=FONT_TITLE, fontweight='bold', pad=8)
ax1.tick_params(axis='both', which='major', labelsize=FONT_TICK)
ax1.grid(axis='y', linestyle=':', alpha=0.6)

ax1.axhline(y=24, color='#e74c3c', linestyle='--', linewidth=1.0)
ax1.text(0.02, 28, '24h Practical Limit', color='#e74c3c', fontsize=FONT_TICK, fontweight='bold')

labels = ['1.86 h', '15.6 h', '8.22 h', '17.1 days', '4.7 years']
for bar, label in zip(bars, labels):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.35, label,
             ha='center', va='bottom', fontsize=7.0, fontweight='bold')

ax2 = fig.add_subplot(1, 2, 2, polar=True)

labels_radar = [
    'Outer-Loop\nScalability',
    'Zero-Forecast\nRobustness',
    'Cross-Config\nGeneralization',
    'Real-Time\nInference Speed',
    'Safety Constraint\nEnforcement'
]
num_vars = len(labels_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

drl_scores = [5, 5, 5, 5, 5]
drl_scores += drl_scores[:1]

mpc_scores = [3, 2, 2, 2, 4]
mpc_scores += mpc_scores[:1]

dp_scores = [1, 1, 1, 3, 5]
dp_scores += dp_scores[:1]

ax2.plot(angles, drl_scores, color='#2ca02c', linewidth=1.6, label='Proposed DRL')
ax2.fill(angles, drl_scores, color='#2ca02c', alpha=0.2)

ax2.plot(angles, mpc_scores, color='#d62728', linewidth=1.4, linestyle='--', label='MPC')
ax2.fill(angles, mpc_scores, color='#d62728', alpha=0.1)

ax2.plot(angles, dp_scores, color='#1f77b4', linewidth=1.4, linestyle=':', label='DP')
ax2.fill(angles, dp_scores, color='#1f77b4', alpha=0.08)

ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=7.2, fontweight='bold')
ax2.set_ylim(0, 5)
ax2.set_yticks([1, 2, 3, 4, 5])
ax2.set_yticklabels(['', '', '', '', ''], fontsize=6)
ax2.set_title('(b) Co-Design Performance Radar', fontsize=FONT_TITLE, fontweight='bold', pad=14)
ax2.legend(loc='lower right', bbox_to_anchor=(1.35, -0.15), fontsize=FONT_LEGEND, frameon=True, edgecolor='#e0e0e0')

plt.subplots_adjust(left=0.09, right=0.88, top=0.86, bottom=0.15, wspace=0.35)
plt.savefig('codesign_computational_comparison.png', dpi=300)
plt.show()