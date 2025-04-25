import numpy as np

# 指标2：早期进展比例
def early_progress_c(time_list, val_list, lobj):
    n = len(val_list)
    early_index = max(1, int(0.1 * n))
    early_val = val_list[early_index]
    final_val = val_list[-1]
    initial_val = val_list[0]

    total_gap = abs(final_val - initial_val)
    early_gap = abs(early_val - initial_val)

    progress_ratio = early_gap / total_gap if total_gap != 0 else 1
    return 1 - progress_ratio  # 越大越好，反转后越小越好
