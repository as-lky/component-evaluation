import optuna
import os
import subprocess
import math
import re
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="receive optimize instruction")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "SC", "MIKS", "MIKSC"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--type", type=str, required=True, choices=["easy", "medium", "hard"], help="the task type")

#parser.add_argument("--train_data_dir", type=str, required=True, help="the train instances input folder")
args = parser.parse_args()

def calc_api(lis):
    # lis 为一个list 其元素为一个list，对应一个点
    if os.path.exists('calc/hbda/build/nonincremental/tmp.txt'):
        os.remove('calc/hbda/build/nonincremental/tmp.txt')
    SUM = 0
    with open('calc/hbda/build/nonincremental/tmp.txt', 'w') as f:
        for i in lis:
            for j in i:
                w = round(j, 8)
                f.write(f"{w} ")
            f.write("\n")
    if os.path.exists('calc/hbda/build/nonincremental/result.txt'):
        os.remove('calc/hbda/build/nonincremental/result.txt')
    
    subprocess.run(['./calc/hbda/build/nonincremental/nonincremental', '-O', 'calc/hbda/build/nonincremental/tmp.txt', '-S', 'calc/hbda/build/nonincremental/result.txt'])    
    with open('calc/hbda/build/nonincremental/result.txt', 'r') as f:
        lines = f.readlines()
        line_result = lines[1].strip()
        result = line_result.split()
        SUM = float(result[-1])
    return SUM

def gapstart_c(time_list, val_list, lobj): # 初始解gap 
    val = val_list[0]
    gap = abs(val - lobj) / lobj if val != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * gap * 100) 
    if t >= 0.99:
        t = 0.99
    return gap, t # 越小越好

def gapend_c(time_list, val_list, lobj): # 最终gap
    val = val_list[-1]
    gap = abs(val - lobj) / lobj if val != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * gap * 100) 
    if t >= 0.99:
        t = 0.99
    return gap, t # 越小越好


def ir_c(time_list, val_list, lobj): # 改进比率
    ir = abs(val_list[-1] - val_list[0]) / val_list[0] if val_list[0] != 0 else 99990
    if ir > 99990:
        ir = 99990
    # 差距可能很大！
    a = math.log10(ir + 10) # 1 到 5
    a = 1 - a / 5
    return ir, a # 越小越好

def nr_c(time_list, val_list, lobj): # 求解的有效率
    num = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.1:
            num += 1
    nr = num / len(val_list) 
    if num == 0: 
        nr = 0.0001
    return num / len(val_list), 1 - nr # 越小越好

def yxtime_c(time_list, val_list, lobj): # 第一个有效解和特优解的时间
    cnt = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.1:
            break
        cnt += 1
    ans1 = time_list[cnt] if cnt < len(time_list) else 99999990
    ans1g = math.log10(10 + ans1) # 1 到 8
    ans1g = ans1g / 20
    cnt = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.01:
            break
        cnt += 1
    ans2 = time_list[cnt] if cnt < len(time_list) else 99999990
    ans2g = math.log10(10 + ans2) # 1 到 8
    ans2g = ans2g / 20
    return ans1 + ans2, ans1g + ans2g # 越小越好
  
# 早期(20% time)进展比例
def early_progress_c(time_list, val_list, lobj):
    th = time_list[-1] * 0.2
    early = -1
    for i in range(len(time_list)):
        if time_list[i] < th:
            early = i
    if early == -1:
        return -1, 0.99
    
    early_val = val_list[early]
    early_gap = abs(early_val - lobj) / lobj if lobj != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * early_gap * 100) 
    if t >= 0.99:
        t = 0.99
    return early_gap, t # 越小越好

# 中期(60% time)进展比例
def medium_progress_c(time_list, val_list, lobj):
    th = time_list[-1] * 0.6
    medium = -1
    for i in range(len(time_list)):
        if time_list[i] < th:
            medium = i
    if medium == -1:
        return -1, 0.99
    
    medium_val = val_list[medium]
    medium_gap = abs(medium_val - lobj) / lobj if lobj != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * medium_gap * 100) 
    if t >= 0.99:
        t = 0.99
    return medium_gap, t # 越小越好

def overall_efficiency_c(time_list, val_list, lobj):
    total_time = time_list[-1] - time_list[0]
    total_improvement = abs(val_list[-1] - val_list[0])
    total_improvement = total_improvement / lobj if lobj != 0 else 9999999999
    if total_improvement == 0:
        return 999999999, 0.999  # 无改进为差的情况
    efficiency = total_time / total_improvement
    if efficiency > 1e4 - 10:
        efficiency = 1e4 - 10
        
    # 假设常见效率范围在 [0.001, 1000]，取log映射
    efficiency_log = math.log10(10 + efficiency)  # 范围 [1, 4]
    return efficiency, (efficiency_log - 1) / 4  # 越小越好

def area_under_curve_c(time_list, val_list, lobj):
    if len(time_list) < 2:
        return -1, 0.5 # 返回中间值
    if lobj == 0:
        normalized_vals = [abs(v) for v in val_list]
    else:
        normalized_vals = [abs(v - lobj) / abs(lobj) for v in val_list]
    
    auc = 0.0
    for i in range(1, len(time_list)):
        delta_t = time_list[i] - time_list[i-1]
        avg_gap = (normalized_vals[i] + normalized_vals[i-1]) / 2
        auc += avg_gap * delta_t
    
    if auc > 1e4 - 10:
        auc = 1e4 - 10
    log_auc = math.log10(10 + auc)  # 1 到 4
    return auc, (log_auc - 1) / 4  # 越小越好

def stagnation_time_c(time_list, val_list, lobj):
    if len(time_list) < 2:
        return -1, 0.5 # 返回中间值
    all_time = 0
    for i in range(len(time_list)-1):
        if val_list[i] == val_list[i+1]:
            all_time += time_list[i+1] - time_list[i]
    all_time /= time_list[-1]
    return all_time, all_time # 越小越好
  
# def imprate_c(time_list, val_list, lobj):
#     if len(val_list) < 2:
#         return 0.5  # 数据太少，返回中等值
#     total_time = time_list[-1] - time_list[0]
#     improve = abs(val_list[0] - val_list[-1])
#     rate = improve / total_time
#     norm_rate = 1 / (rate + 1e-5)  # 改进越快值越小
#     norm_rate = min(norm_rate, 10)
#     return norm_rate / 10  # 越小越好

# def stability_c(time_list, val_list, lobj):
#     diffs = [abs(val_list[i+1] - val_list[i]) for i in range(len(val_list)-1)]
#     if not diffs:
#         return 0
#     ss = 0
#     for diff in diffs:
#         ss += diff
#     avg_diff = ss / len(diffs)
#     norm_diff = min(avg_diff / (abs(val_list[0]) + 1e-6), 1)
#     return norm_diff  # 越小越好

# def earlygap_c(time_list, val_list, lobj):
#     total_time = time_list[-1]
#     threshold_time = total_time * 0.1
#     early_vals = [val_list[i] for i in range(len(time_list)) if time_list[i] <= threshold_time]
#     if not early_vals:
#         early_vals = [val_list[0]]
#     gaps = [abs(v - lobj) / v if v != 0 else 1 for v in early_vals]
#     ss = 0
#     for gap in gaps:
#         ss += gap
#     mean_gap = ss / len(gaps)
#     return min(mean_gap, 1)  # 越小越好

# def trend_c(time_list, val_list, lobj):
#     if len(val_list) < 3:
#         return 0.5  # 数据太少，返回中等值

#     # 统计每一步是否改进了
#     improvements = 0
#     for i in range(1, len(val_list)):
#         if val_list[i] < val_list[i - 1]:
#             improvements += 1
#     trend_ratio = improvements / (len(val_list) - 1)

#     # 越接近持续改进，trend_ratio 越接近 1，越好，取反归一化
#     return 1 - trend_ratio  # 越小越好

#TODO:不严格单调的噢!
#TODO:求解的稳定性指标?

def calc(data, lobj, type):
    # 最终gap
    # TODO: 做数学推导
    result_list = data['result_list']
    time_list = [_[0] for _ in result_list]
    val_list = [_[1] for _ in result_list]

    if lobj > val_list[0]:
        lobj = max(lobj, val_list[-1])
    else:
        lobj = min(lobj, val_list[-1])

    if type == 'easy':
        threshold = 100
    elif type == 'medium':
        if args.taskname == 'IS':
            threshold = 600
        if args.taskname == 'MIKS' or args.taskname == 'MIKSC':
            threshold = 4000
        if args.taskname == 'SC':
            threshold = 600
        if args.taskname == 'MVC':
            threshold = 600
    elif type == 'hard':
        if args.taskname == 'IS':
            threshold = 3500
        if args.taskname == 'MIKS' or args.taskname == 'MIKSC':
            threshold = 8000
        if args.taskname == 'SC':
            threshold = 3500
        if args.taskname == 'MVC':
            threshold = 1800
    else:
        threshold = -1
    if time_list[-1] < threshold:
        time_list.append(threshold)
        val_list.append(val_list[-1])
    gapstartori, gapstart = gapstart_c(time_list, val_list, lobj)
    gapendori, gapend = gapend_c(time_list, val_list, lobj)
    irori, ir = ir_c(time_list, val_list, lobj)
    nrori, nr = nr_c(time_list, val_list, lobj)
#    sgapori, stimeori, sgap, stime = sgap_stime_c(time_list, val_list, lobj)
#    stimeori, stime = stime_c(time_list, val_list, lobj)
    yxtimeori, yxtime = yxtime_c(time_list, val_list, lobj)
    early_progressori, early_progress = early_progress_c(time_list, val_list, lobj)
    medium_progressori, medium_progress = medium_progress_c(time_list, val_list, lobj)
    area_under_curveori, area_under_curve = area_under_curve_c(time_list, val_list, lobj)
    stagnation_timeori, stagnation_time = stagnation_time_c(time_list, val_list, lobj)
    
    ll0 = [
        gapstartori,
        gapendori,
#        irori, 依赖初始解
        nrori,
#        stimeori, 
        yxtimeori,
     #   sgapori,
        early_progressori,
        medium_progressori,
  #      overall_efficiencyori, 依赖初始解
        area_under_curveori,
#        stagnation_timeori,
    ]
    
    ll1 = [ gapstart,
            gapend,
 #           ir,  依赖初始解
            nr,
#            stime,
            yxtime,
            early_progress,
            medium_progress,
 #           overall_efficiency, 依赖初始解
            area_under_curve,
#            stagnation_time,
      #      sgap,
#          imprate_c(time_list, val_list, lobj),
#          stability_c(time_list, val_list, lobj),
#          earlygap_c(time_list, val_list, lobj),
#          trend_c(time_list, val_list, lobj),
          # TODO: 使用大模型指标
          ]
#    ll2 = [0.05, 0.4, 0.15, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05]
 #   for i in range(len(ll1)):
#        ll1[i] = 1 - (1 - ll1[i]) * ll2[i] / 0.4
 #       print(ll1[i])
 #       ll1[i] /= 2
    return ll0, ll1


    # TODO:加入更多指标
    # 几个收敛指标?
    # 稳定性指标？
    # 加入大模型的
    ...
    
def work_gurobi(instance):
    instance_name = os.path.basename(instance)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
   
    if not os.path.exists(f'./logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'):
        rt = 3000
        if args.type == 'medium':
            rt = 12000
        if args.type == 'hard':
            rt = 30000
        subprocess.run(["python", "main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
        "--graphencode", "test", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", f"{rt}"])    
    
    des = f'./logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # 最大化
        return data['obj'] * (1 + data['gap'] / 100), -1
    else:
        return data['obj'] * (1 - data['gap'] / 100), 1

NUM_GPUS = 4

#lobj, _ = work_gurobi(args.instance_path)
#lobj, _ = 55994, -1
#lobj, _ = 563956, -1
#lobj, _ = 25268, -1 # IS medium8
lobj, _ = 242, 1 # SC medium2
#lobj, _ = 252240, -1 # IS hard0
#lobj, _ = 246449, 1 # MVC hard0
#lobj, _ = 5765, 1 # MVC fakemedium3
#lobj, _ = 839653, 1 # SC fakemedium0
#lobj, _ = 2624, 1# SC hard2
#lobj, _ = 162, 1# MIKSC hard0
#lobj, _ = 11932, -1# MIKSC fakemedium5
#lobj, _ = 75221, -1# MIKSC fakehard1

#lobj, _ = 24575, 1 # MVC medium5
#lobj, _ = 16.52, 1 # MIKSC medium5  bir_gcn_nr_ACP_

def objective(trial):
#    gpu_id = trial.number % NUM_GPUS
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    #block = trial.suggest_int('block', 1, 5)
    #ratio = trial.suggest_float('max_turn_ratio', 0.1, 0.5)
    type_ = args.type
#    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'SC', '--instance_path', './Dataset/SC_medium_instance/SC_medium_instance/LP/SC_medium_instance_0.lp', 
#            '--graphencode', 'bi', '--predict', 'gat', '--whole_time_limit', '600', '--modify', 'nr', '--search', 'ACP']
    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'MVC', '--instance_path', './Dataset/MVC_medium_instance/MVC_medium_instance/LP/MVC_medium_instance_0.lp', 
            '--graphencode', 'default', '--predict', 'gurobi', '--whole_time_limit', '100', '--modify', 'default', '--search', 'ACP']

    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'MIKS', '--instance_path', './Dataset/MIKS_fakemedium_instance/MIKS_fakemedium_instance/LP/MIKS_fakemedium_instance_0.lp', 
            '--graphencode', 'bi', '--predict', 'gat', '--whole_time_limit', '2000', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cuda:1', '--taskname', 'MIKS', '--instance_path', './Dataset/MIKS_fakemedium_instance/MIKS_fakemedium_instance/LP/MIKS_fakemedium_instance_0.lp', 
            '--graphencode', 'bi', '--predict', 'gcn', '--whole_time_limit', '4000', '--modify', 'sr', '--search', 'MIH']
    
    exec = ['python', 'main.py', '--device', 'cuda:1', '--taskname', 'IS', '--instance_path', './Dataset/IS_easy_instance/IS_easy_instance/LP/IS_easy_instance_0.lp', 
            '--graphencode', 'bi', '--predict', 'gcn', '--whole_time_limit', '300', '--modify', 'sr', '--search', 'MIH']
    
    exec = ['python', 'main.py', '--device', 'cuda', '--taskname', 'IS', '--instance_path', './Dataset/IS_fakemedium_instance/IS_fakemedium_instance/LP/IS_fakemedium_instance_10.lp', 
            '--graphencode', 'trir', '--predict', 'gcn', '--whole_time_limit', '600', '--modify', 'nr', '--search', 'NALNS']
    
    exec = ['python', 'main.py', '--device', 'cuda:1', '--taskname', 'IS', '--instance_path', './Dataset/IS_medium_instance/IS_medium_instance/LP/IS_medium_instance_8.lp', 
            '--graphencode', 'trir', '--predict', 'gcn', '--whole_time_limit', '600', '--modify', 'nr', '--search', 'NALNS']
    
    exec = ['python', 'main.py', '--device', 'cuda:3', '--taskname', 'SC', '--instance_path', './Dataset/SC_medium_instance/SC_medium_instance/LP/SC_medium_instance_2.lp', 
            '--graphencode', 'default', '--predict', 'gurobi', '--whole_time_limit', '600', '--modify', 'default', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cpu', '--taskname', 'IS', '--instance_path', './Dataset/IS_hard_instance/IS_hard_instance/LP/IS_hard_instance_0.lp', 
            '--graphencode', 'trir', '--predict', 'gcn', '--whole_time_limit', '3500', '--modify', 'nr', '--search', 'NALNS']
    
    exec = ['python', 'main.py', '--device', 'cpu', '--taskname', 'MVC', '--instance_path', './Dataset/MVC_hard_instance/MVC_hard_instance/LP/MVC_hard_instance_0.lp', 
           '--graphencode', 'bi', '--predict', 'gcn', '--whole_time_limit', '3500', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cuda:1', '--taskname', 'MVC', '--instance_path', './Dataset/MVC_fakemedium_instance/MVC_fakemedium_instance/LP/MVC_fakemedium_instance_3.lp', 
           '--graphencode', 'bi', '--predict', 'gcn', '--whole_time_limit', '600', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'SC', '--instance_path', './Dataset/SC_fakemedium_instance/SC_fakemedium_instance/LP/SC_fakemedium_instance_0.lp', 
            '--graphencode', 'bi', '--predict', 'gat', '--whole_time_limit', '600', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'SC', '--instance_path', './Dataset/SC_hard_instance/SC_hard_instance/LP/SC_hard_instance_2.lp', 
            '--graphencode', 'bi', '--predict', 'gat', '--whole_time_limit', '3500', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cpu', '--taskname', 'MIKSC', '--instance_path', './Dataset/MIKSC_hard_instance/MIKSC_hard_instance/LP/MIKSC_hard_instance_0.lp', 
            '--graphencode', 'bir', '--predict', 'gcn', '--whole_time_limit', '8000', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cuda:1', '--taskname', 'MIKSC', '--instance_path', './Dataset/MIKSC_fakemedium_instance/MIKSC_fakemedium_instance/LP/MIKSC_fakemedium_instance_5.lp', 
            '--graphencode', 'bir', '--predict', 'gcn', '--whole_time_limit', '4000', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cpu', '--taskname', 'MIKSC', '--instance_path', './Dataset/MIKSC_fakehard_instance/MIKSC_fakehard_instance/LP/MIKSC_fakehard_instance_1.lp', 
            '--graphencode', 'bir', '--predict', 'gcn', '--whole_time_limit', '8000', '--modify', 'nr', '--search', 'ACP']
    
    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'SC', '--instance_path', './Dataset/SC_medium_instance/SC_medium_instance/LP/SC_medium_instance_2.lp', 
            '--graphencode', 'bir', '--predict', 'gat', '--whole_time_limit', '600', '--modify', 'nr', '--search', 'ACP']
    
    
    #exec = ['python', 'main.py', '--
    # device', 'cuda:2', '--taskname', 'MVC', '--instance_path', './Dataset/MVC_medium_instance/MVC_medium_instance/LP/MVC_medium_instance_5.lp', 
    #        '--graphencode', 'bi', '--predict', 'gcn', '--whole_time_limit', '600', '--modify', 'nr', '--search', 'ACP']
    
    
    # exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'IS', '--instance_path', './Dataset/IS_fakemedium_instance/IS_fakemedium_instance/LP/IS_fakemedium_instance_10.lp', 
    #         '--graphencode', 'trir', '--predict', 'gcn', '--whole_time_limit', '600', '--modify', 'np', '--search', 'ACP']

    # exec = ['python', 'main.py', '--device', 'cpu', '--taskname', 'IS', '--instance_path', './Dataset/IS_fakehard_instance/IS_fakehard_instance/LP/IS_fakehard_instance_0.lp', 
    #         '--graphencode', 'trir', '--predict', 'gcn', '--whole_time_limit', '3500', '--modify', 'nr', '--search', 'NALNS']
    
#    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'MIKSC', '--instance_path', './Dataset/MIKSC_medium_instance/MIKSC_medium_instance/LP/MIKSC_medium_instance_5.lp', 
 #           '--graphencode', 'bir', '--predict', 'gcn', '--whole_time_limit', '4000', '--modify', 'nr', '--search', 'ACP']
    
    
    
    block = trial.suggest_int('block', 2, 10)   
    ratio = trial.suggest_float('ratio', 0.1, 0.9)
    
    exec += ['--search_ACP_LNS_block', str(block), '--search_ACP_LNS_max_turn_ratio', str(ratio)]
 
#    choose = trial.suggest_float('choose', 0.1, 0.9)
#    set_pa = trial.suggest_float('set_pa', 0.1, 0.9)
    
#    exec += ['--search_LIH_MIH_NALNS_choose', str(choose), '--search_LIH_MIH_set_pa', str(set_pa)]
#    exec += ['--search_LIH_MIH_NALNS_choose', str(choose)]
    
    subprocess.run(exec)
    
    instance_name = os.path.basename(args.instance_path)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)


    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
 #   we = f"default_gurobi_default_ACP_{block}_{ratio}_"
#    we = f"bi_gcn_sr_MIH_{choose}_{set_pa}_"
#    we = f"trir_gcn_nr_NALNS_{choose}_" # IS hard

#    we = f"trir_gcn_np_ACP_{block}_{ratio}_"
#    we = f"bi_gcn_sr_ACP_{block}_{ratio}_" # SC best
    we = f"bir_gat_nr_ACP_{block}_{ratio}_" # SC cibest with gat
#    we = f"bir_gcn_nr_ACP_{block}_{ratio}_" # MIKSC best
#    we = f"default_gurobi_default_ACP_{block}_{ratio}_"
#    we = f"bi_gcn_nr_ACP_{block}_{ratio}_" # MVC best
#    we = f"bi_gcn_nr_ACP_{block}_{ratio}_" # MVC best
#    we = f"bi_gat_nr_ACP_{block}_{ratio}_" # SC cibest
#    we = f"bir_gcn_nr_ACP_{block}_{ratio}_" # MIKSC cibest


    des = f'./logs/work/{args.taskname}/{we}/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    LISORI, LIS = calc(data, lobj, type_)
    result___ = calc_api([LIS]) * 1e7
    with open(f'./{args.taskname}_tmp.txt', 'a') as f:
#        f.write(f"{result___} {choose} {set_pa} {LISORI} {LIS}\n")
#        f.write(f"{result___} {choose} {LISORI} {LIS}\n")
        f.write(f"{result___} {block} {ratio} {LISORI} {LIS}\n")

    return result___

#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/ISmedium", study_name="ISmediumNALNS")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/SCmedium", study_name="SCmediumgurobiACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/MIKSCmedium", study_name="MIKSCmediumACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/IShard", study_name="IShardNALNS")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/MVCmedium", study_name="MVCmediumACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/MVChard", study_name="MVChardACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/MVCfakemedium", study_name="MVCfakemediumACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/SCfakemedium", study_name="SCfakemediumgatACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/SChard", study_name="SChardgatACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/MIKSChard", study_name="MIKSChardACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/MIKSCfakemedium", study_name="MIKSCfakemediumACP")
#study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/MIKSCfakehard", study_name="MIKSCfakehardACP")

study = optuna.load_study(storage="postgresql://luokeyun:lky883533600@localhost:5432/SCmedium", study_name="SCmediumgatACP")

#study.optimize(objective, n_trials=2, n_jobs=2)  # 并行4个worker（=4块GPU）
study.optimize(objective, n_trials=5)


# exec  参数   we  输出  study