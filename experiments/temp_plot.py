import matplotlib.pyplot as plt

# new fig
targets = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
recalls = [0.65671, 0.76326, 0.82383, 0.89361, 0.93466, 0.96498]
activations = [0.14023, 0.20221, 0.25769, 0.3612, 0.46924, 0.60281]
fig, (ax_top, ax_bottom) = plt.subplots(
	2,
	1,
	sharex=True,
	gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
)
ax_top.plot(targets, recalls, marker='o')
ax_top.plot(targets, targets, linestyle='--', color='gray')  # add y=x line for reference
ax_top.set_ylabel('Theoretical Recall@10')
ax_top.set_axisbelow(True)
ax_top.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8, color='0.6', alpha=0.9)
ax_bottom.plot(targets, activations, marker='o')
ax_bottom.set_xlabel('Recall Target')
ax_bottom.set_ylabel('Activation')
ax_bottom.set_axisbelow(True)
ax_bottom.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8, color='0.6', alpha=0.9)
ax_top.set_title('MSTuring-30M-clustered, 10 partitions')
plt.savefig('recall_target_vs_recall.png')

# 


nprobes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
recalls = [0.54201, 0.72257, 0.81883, 0.87856, 0.92072, 0.94934, 0.96998, 0.9845, 0.99443]
activations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

fig, (ax_top, ax_bottom) = plt.subplots(
	2,
	1,
	sharex=True,
	gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
)
ax_top.plot(nprobes, recalls, marker='o')
ax_top.set_ylabel('Theoretical Recall@10')
ax_top.set_axisbelow(True)
ax_top.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8, color='0.6', alpha=0.9)
ax_bottom.plot(nprobes, activations, marker='o')
ax_bottom.set_xlabel('nprobe')
ax_bottom.set_ylabel('Activation')
ax_bottom.set_axisbelow(True)
ax_bottom.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8, color='0.6', alpha=0.9)
# save plot
ax_top.set_title('MSTuring-30M-clustered, 10 partitions')
plt.savefig('nprobe_vs_recall.png')

branching_factors = [1, 2, 5, 15, 20, 25, 30]
recalls = [0.54209, 0.67659, 0.81492, 0.92549, 0.94393, 0.95622, 0.96428]
activations = [0.1, 0.1562, 0.26439, 0.46194, 0.52568, 0.57792, 0.62076]
fig, (ax_top, ax_bottom) = plt.subplots(
	2,
	1,
	sharex=True,
	gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
)
ax_top.plot(branching_factors, recalls, marker='o')
ax_top.set_ylabel('Theoretical Recall@10')
ax_top.set_axisbelow(True)
ax_top.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8, color='0.6', alpha=0.9)
ax_bottom.plot(branching_factors, activations, marker='o')
ax_bottom.set_xlabel('Branching Factor')
ax_bottom.set_ylabel('Activation')
ax_bottom.set_axisbelow(True)
ax_bottom.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8, color='0.6', alpha=0.9)
ax_top.set_title('MSTuring-30M-clustered, 10 partitions')
plt.savefig('branching_factor_vs_recall.png')


# branching_factor param=1 recall=0.54209 activation=0.1 imbalance=0.304117 time_s=0.093338
# branching_factor param=2 recall=0.67659 activation=0.1562 imbalance=0.308194 time_s=0.0601284
# branching_factor param=5 recall=0.81492 activation=0.26439 imbalance=0.280042 time_s=0.0542979
# branching_factor param=15 recall=0.92549 activation=0.46194 imbalance=0.223767 time_s=0.0458091
# branching_factor param=20 recall=0.94393 activation=0.52568 imbalance=0.204425 time_s=0.0425962
# branching_factor param=25 recall=0.95622 activation=0.57792 imbalance=0.186606 time_s=0.047036
# branching_factor param=30 recall=0.96428 activation=0.62076 imbalance=0.174045 time_s=0.043326
# Finished computing routing metrics for branching factor strategy
# recall_target param=0.5 recall=0.65671 activation=0.14023 imbalance=0.414583 time_s=0.0432187
# recall_target param=0.7 recall=0.76326 activation=0.20221 imbalance=0.390164 time_s=0.0435315
# recall_target param=0.8 recall=0.82383 activation=0.25769 imbalance=0.366707 time_s=0.0435438
# recall_target param=0.9 recall=0.89361 activation=0.3612 imbalance=0.316417 time_s=0.043881
# recall_target param=0.95 recall=0.93466 activation=0.46924 imbalance=0.267841 time_s=0.043305
# recall_target param=0.98 recall=0.96498 activation=0.60281 imbalance=0.207504 time_s=0.043331
# Finished computing routing metrics for recall target strategy
# nprobe param=1 recall=0.54201 activation=0.1 imbalance=0.303762 time_s=0.0997021
# nprobe param=2 recall=0.72257 activation=0.2 imbalance=0.28884 time_s=0.0987281
# nprobe param=3 recall=0.81883 activation=0.3 imbalance=0.271304 time_s=0.098401
# nprobe param=4 recall=0.87856 activation=0.4 imbalance=0.247358 time_s=0.0998847
# nprobe param=5 recall=0.92072 activation=0.5 imbalance=0.217744 time_s=0.105554
# nprobe param=6 recall=0.94934 activation=0.6 imbalance=0.184385 time_s=0.103405
# nprobe param=7 recall=0.96998 activation=0.7 imbalance=0.148311 time_s=0.0986057
# nprobe param=8 recall=0.9845 activation=0.8 imbalance=0.107226 time_s=0.0989897
# nprobe param=9 recall=0.99443 activation=0.9 imbalance=0.0594766 time_s=0.0990512