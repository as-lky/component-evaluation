# Llama-4
import math
from scipy.interpolate import PchipInterpolator

def monotonicity_c(time_list, val_list, lobj):  # 目标函数值和时间单调性
    """
    评价目标函数值和时间单调性
    """
    # 检查目标函数值是否单调
    val_monotonic = all(val_list[i] <= val_list[i + 1] for i in range(len(val_list) - 1)) or all(val_list[i] >= val_list[i + 1] for i in range(len(val_list) - 1))
    # 检查时间是否单调
    time_monotonic = all(time_list[i] <= time_list[i + 1] for i in range(len(time_list) - 1))
    # 计算单调性指标
    if val_monotonic and time_monotonic:
        return 0
    else:
        return min(1, max(0, 1 - (sum([val_monotonic, time_monotonic]) / 2)))

def final_gap_c(time_list, val_list, lobj):  # 最终差距
    """
    评价最终差距
    """
    val = val_list[-1]
    gap = abs(val - lobj) / val if val != 0 else 1000
    return min(gap / 1000, 1)  # 越小越好

def convergence_rate_c(time_list, val_list, lobj):  # 收敛速度
    """
    评价收敛速度
    """
    if len(val_list) < 2:
        return 1
    vv = val_list
    for i in range(1, len(vv)):
        vv[i] = max(vv[i], vv[i - 1] + 1e-8)

    pchip = PchipInterpolator(val_list, time_list)
    stime = pchip(lobj)
    if stime > 1e6:
        stime = 1e6
    if stime < 0.1:
        stime = 0.1
    stime = math.log(stime)  # -1 到 6
    return (stime + 1) / 7  # 越小越好

def oscillation_c(time_list, val_list, lobj):  # 振荡程度
    """
    评价振荡程度
    """
    if len(val_list) < 2:
        return 0
    oscillation = sum(1 for i in range(1, len(val_list)) if (val_list[i] - val_list[i - 1]) * (val_list[i - 1] - val_list[0]) < 0) / (len(val_list) - 1)
    return oscillation  # 越小越好

def time_efficiency_c(time_list, val_list, lobj):  # 时间效率
    """
    评价时间效率
    """
    time = time_list[-1]
    if time > 1e6:
        time = 1e6
    if time < 0.1:
        time = 0.1
    time = math.log(time)  # -1 到 6
    return (time + 1) / 7  # 越小越好






#deepseek
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

def relative_gap_improvement(time_list, val_list, lobj):
    """
    相对gap改进率：衡量从初始到最终的相对改进程度
    原始指标：(initial_gap - final_gap)/initial_gap
    映射：log变换后线性归一化
    """
    initial_gap = abs(val_list[0] - lobj)/abs(lobj) if lobj != 0 else 1e6
    final_gap = abs(val_list[-1] - lobj)/abs(lobj) if lobj != 0 else 1e6
    
    if initial_gap < 1e-6:  # 已经最优
        return 0.0
    
    improvement = (initial_gap - final_gap) / initial_gap
    improvement = max(improvement, 1e-6)  # 避免log(0)
    
    # 经验值：改进率通常在1e-6到1之间
    log_imp = -math.log10(improvement)  # 映射到6到0
    return min(log_imp / 6, 0.999)  # 0表示完全改进，接近1表示几乎没有改进

def area_under_curve(time_list, val_list, lobj):
    """
    曲线下面积：衡量收敛速度和质量
    原始指标：归一化的目标值与最优值的差距在时间上的积分
    映射：log变换后线性归一化
    """
    if lobj == 0:
        normalized_vals = [abs(v) for v in val_list]
    else:
        normalized_vals = [abs(v - lobj)/abs(lobj) for v in val_list]
    
    # 计算曲线下面积（梯形法）
    auc = 0.0
    for i in range(1, len(time_list)):
        delta_t = time_list[i] - time_list[i-1]
        avg_gap = (normalized_vals[i] + normalized_vals[i-1]) / 2
        auc += avg_gap * delta_t
    
    # 经验值：典型范围从1e-4到1e4
    if auc < 1e-6:
        return 0.0
    log_auc = math.log10(auc)  # -4到4
    return (log_auc + 4) / 8  # 映射到[0,1)

def time_to_90percent(time_list, val_list, lobj):
    """
    达到90%改进的时间：衡量收敛速度
    原始指标：达到初始和最优值之间90%改进的时间
    映射：log时间后线性归一化
    """
    if len(val_list) < 2:
        return 1.0
    
    initial_val = val_list[0]
    target = lobj + 0.9 * (initial_val - lobj) if initial_val > lobj else lobj - 0.9 * (lobj - initial_val)
    
    # 找到首次达到或超过目标的时间
    for i, val in enumerate(val_list):
        if (initial_val > lobj and val <= target) or (initial_val < lobj and val >= target):
            t = time_list[i]
            break
    else:
        t = time_list[-1]
    
    # 经验值：时间范围从0.1秒到1e6秒
    if t < 0.1:
        return 0.0
    log_t = math.log10(t)  # -1到6
    return min((log_t + 1) / 7, 0.999)

def consistency_c(time_list, val_list, lobj):
    """
    一致性：衡量解的稳定改进程度
    原始指标：违反单调改进的次数占总步数的比例
    映射：直接使用比例，但反转(1-x)
    """
    violations = 0
    for i in range(1, len(val_list)):
        if (val_list[i] > val_list[i-1] and val_list[-1] < lobj) or (val_list[i] < val_list[i-1] and val_list[-1] > lobj):
            violations += 1
    
    ratio = violations / (len(val_list) - 1) if len(val_list) > 1 else 1.0
    return ratio  # 0表示完全一致，接近1表示频繁震荡

def early_convergence_c(time_list, val_list, lobj):
    """
    早期收敛：衡量算法早期收敛能力
    原始指标：在前10%时间内达到的最终gap比例
    映射：线性映射
    """
    if len(val_list) < 2:
        return 1.0
    
    cutoff_time = time_list[-1] * 0.1
    cutoff_idx = 0
    while cutoff_idx < len(time_list) and time_list[cutoff_idx] <= cutoff_time:
        cutoff_idx += 1
    
    if cutoff_idx == 0:
        return 1.0
    
    early_gap = abs(val_list[cutoff_idx-1] - lobj)/abs(lobj) if lobj != 0 else 1e6
    final_gap = abs(val_list[-1] - lobj)/abs(lobj) if lobj != 0 else 1e6
    
    if final_gap < 1e-6:  # 已经最优
        return 0.0
    
    ratio = early_gap / final_gap
    return min(ratio, 1.0)  # 1表示早期没有进展，0表示早期就完全收敛




# GPT 4.5
# 指标1：收敛稳定性
def stability_c(time_list, val_list, lobj):
    diffs = np.diff(val_list)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    stability_ratio = sign_changes / (len(val_list) - 2)
    return min(stability_ratio, 0.999)  # 越少越稳定，越好

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

# 指标3：中段停滞程度
def mid_stagnation_c(time_list, val_list, lobj):
    n = len(val_list)
    start_mid = int(0.4 * n)
    end_mid = int(0.6 * n)
    mid_range = val_list[start_mid:end_mid+1]

    mid_gap = abs(mid_range[-1] - mid_range[0])
    total_gap = abs(val_list[-1] - val_list[0])

    stagnation_ratio = 1 - (mid_gap / total_gap) if total_gap != 0 else 1
    return min(stagnation_ratio, 0.999)  # 越小越好

# 指标4：最终快速收敛指数
def late_convergence_c(time_list, val_list, lobj):
    n = len(val_list)
    start_late = int(0.8 * n)
    late_range = val_list[start_late:]

    late_gap = abs(late_range[-1] - late_range[0])
    total_gap = abs(val_list[-1] - val_list[0])

    convergence_ratio = late_gap / total_gap if total_gap != 0 else 0
    return 1 - convergence_ratio  # 越大越好，反转后越小越好

# 指标5：综合改进效率
def overall_efficiency_c(time_list, val_list, lobj):
    total_time = time_list[-1] - time_list[0]
    total_improvement = abs(val_list[-1] - val_list[0])

    if total_improvement == 0:
        return 0.999  # 无改进为差的情况
    efficiency = total_time / total_improvement

    # 假设常见效率范围在 [0.001, 1000]，取log映射
    efficiency = min(max(efficiency, 0.001), 1000)
    efficiency_log = math.log10(efficiency)  # 范围 [-3, 3]
    efficiency_normalized = (efficiency_log + 3) / 6  # 映射到[0,1]
    return min(efficiency_normalized, 0.999)  # 越小越好

#
# 
# 指标1：收敛稳定性（Convergence Stability）
# 说明：

# 观察目标值变化的单调程度，如果目标值波动较大（虽然整体单调但局部波动频繁），则稳定性差。
# 稳定性越好，映射值越接近0。
# 实现：

# 计算val_list的前后差分，若差分符号变化频繁，稳定性差。
# 用符号变化次数占比，映射到[0,1)。
# 指标2：早期进展比例（Early Progress Rate）
# 说明：

# 评价算法在求解前期（前10%时间）内能达到接近最终目标值的比例。
# 若前期就逼近最终值，得分接近0（优秀）；否则接近1（较差）。
# 指标3：中段停滞程度（Mid-stage Stagnation Rate）
# 说明：

# 检验求解中期（40%-60%时间）目标值变化幅度。
# 若中期几乎无变化，说明中期停滞严重，得分接近1；若中期变化明显，得分接近0。
# 指标4：最终快速收敛指数（Late-stage Convergence Index）
# 说明：

# 考察末段（后20%时间）目标值改善速率。
# 若末段改善显著，得分接近0；否则接近1。
# 指标5：综合改进效率（Overall Improvement Efficiency）
# 说明：

# 考虑整个求解过程中，目标值改进与所耗费的总时间比例。
# 改进幅度大且消耗时间短，得分接近0；反之接近1。
# ##