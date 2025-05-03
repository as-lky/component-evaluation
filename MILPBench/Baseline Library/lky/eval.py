from grapheme import endswith
from sympy import Sum
import torch
import os
import argparse
import time
import re
import subprocess
import json
import shutil
import math
import numpy as np
from scipy.optimize import curve_fit, brentq

# python eval.py --taskname IS --instance_path ./Dataset/IS_easy_instance/IS_easy_instance/LP --train_data_dir ./Dataset/IS_easy_instance/IS_easy_instance/

parser = argparse.ArgumentParser(description="receive evaluate instruction")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "MIKS", "SC"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--train_data_dir", type=str, required=True, help="the train instances input folder")
parser.add_argument("--type", type=str, help="easy medium hard")
parser.add_argument("--task", type=str, help="eval run task")
parser.add_argument("--eval", action="store_true", help="exec eval func")

args = parser.parse_args()

def c(a):
    tmp = os.path.basename(a)
    tmp = re.match(r".*_([0-9]+)", tmp)
    tmp = tmp.group(1)
    return int(tmp) <= 9


def work_gurobi(instance):
    instance_name = os.path.basename(instance)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
   
    if not os.path.exists(f'./logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'):
        subprocess.run(["python", "main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
        "--graphencode", "test", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "3000"])    
    
    des = f'./logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # 最大化
        return data['obj'] * (1 + data['gap'] / 100), -1
    else:
        return data['obj'] * (1 - data['gap'] / 100), 1
    
instance_path = args.instance_path
train_data_dir = args.train_data_dir


if args.task == "birgat":
    grlis = ["bir"]
    prelis = ["gat"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
elif args.task == "bigat":
    grlis = ["bi"]
    prelis = ["gat"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
elif args.task == "gurobisearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi", "scip"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["gurobi"]
elif args.task == "LIHsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi", "scip"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["LIH"]
elif args.task == "MIHsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi", "scip"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["MIH"]
elif args.task == "LNSsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi", "scip"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["LNS"]
elif args.task == "NALNSsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi", "scip"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["NALNS"]
elif args.task == "ACPsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi", "scip"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["ACP"]
elif args.task == "SCIPsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi", "scip"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["scip"]
elif args.task == "cplex":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["cplex"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP", "scip"]
else :
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["cplex", "gcn", "gurobi", "scip", "gat"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP", "scip"]


# grlis = ["bi"]
# prelis = ["gcn"]
# modlis = ["sr"]
# sealis = ["ACP"]

# grlis = ["bi", "bir", "tri", "trir", "default"]
# prelis = ["gcn", "gurobi", "scip"]
# #grlis = ["bi"]
# #prelis = ["gat"]
# modlis = ["sr", "nr", "np", "default"]
# #sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
# sealis = ["ACP"]
#sealis = ["MIH", "LNS", "NALNS"]

instancelis = [os.path.join(instance_path, file) for file in os.listdir(instance_path) if c(file)] # 10 instances

score_dic = {}

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
    gap = abs(val - lobj) / lobj if lobj != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * gap * 100) 
    if t >= 0.95:
        t = 0.95
    return gap, t # 越小越好

def gapend_c(time_list, val_list, lobj): # 最终gap
    val = val_list[-1]
    print(val, lobj)
    gap = abs(val - lobj) / lobj if lobj != 0 else 999999999  
    k = 0.01
    return gap, 1 - math.exp(-k * gap * 100) # 越小越好


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
        if abs(_ - lobj) / lobj < 0.05:
            num += 1
    nr = num / len(val_list) 
    if num == 0: 
        nr = 0.0001
    return num / len(val_list), 1 - nr # 越小越好

def sgap_stime_c(time_list, val_list, lobj): # 预估收敛gap & 收敛到预估收敛gap一定范围内的预估收敛时间
    if len(val_list) < 2:
        return -1, 0.5 # 数据太少，返回中等值
    type = -1 if lobj < val_list[0] else 1
    vv, tt = val_list.copy(), time_list.copy()
    
    threshold = abs(vv[-1] - lobj) / vv[-1] if vv[-1] != 0 else 999999999  
    threshold /= len(time_list) * 5
    threshold = min(threshold, 1e-8)

    vv[0] = abs(vv[0] - lobj) / vv[0] if vv[0] != 0 else 999999999
    for i in range(1, len(vv)):
        gap_tmp = abs(vv[i] - lobj) / vv[i] if vv[i] != 0 else 999999999  
        vv[i] = min(gap_tmp, vv[i - 1] - threshold)

    for i in range(len(vv)):
        vv[i] = -vv[i]
    
    
    GAP = -1
    for i in range(len(tt) - 1):
        if abs(vv[i + 1] - vv[i]) / abs(vv[i + 1]) > 3:
           GAP = i 
    print(val_list, GAP)
    print(vv)

    tt = tt[(GAP + 1):-1]
    vv = vv[(GAP + 1):-1]
    
    # def poly2(x, a, b, c):
    #     return a * x ** 2 + b * x + c

    # def poly3(x, a, b, c, d):
    #     return a * x ** 3 + b * x ** 2 + c * x + d

    # def exp_func(x, a, b, c):
    #     return a * np.exp(b * x) + c

    # def log_func(x, a, b, c):
    #     return a * np.log(b * x) + c

    def func(x, b, c):
        return b * np.exp(c * x)
    popt, _ = curve_fit(func, tt, vv, maxfev=10000, bounds=([-np.inf, -np.inf], [0, np.inf]))
    for i in range(len(tt)):
        print(i, tt[i], vv[i], func(tt[i], *popt))
    print(func(1000, *popt))
    return 0, 0
    # funcs = {
    #     "poly2": poly2,
    #     "poly3": poly3,
    #     "exp": exp_func,
    #     "log": log_func
    # }
    
    # results = {}
    # for name, func in funcs.items():
    #     bounds = (-np.inf, np.inf)
        
    #     if name == "log":
    #         bounds = ([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])
            
    #     popt, _ = curve_fit(func, tt, vv, maxfev=10000, bounds=bounds)
    #     fitted_vals = func(np.array(tt), *popt)
    #     r2 = r2_score(vv, fitted_vals)
    #     results[name] = (func, popt, r2)
    
    # mmax = -9999999
    # best = None
    # for name, _ in results.items():
    #     if mmax < results[name][2]:
    #         mmax = results[name][2]
    #         best = results[name]

    # for i in range(len(tt)):
    #     print(i, tt[i], vv[i], best[0](time_list[i], *best[1]))
    # for i in range(8):
    #     print(i, 10 ** i, best[0](10 ** i, *best[1]))
    #     print(-i, -10 ** i, best[0](-10 ** i, *best[1]))

    # import pandas as pd
    # data = pd.DataFrame({'x': time_list, 'y': vv})
    # reg = setup(data, target='y', session_id=123)
    # best_model = compare_models(fold=5)
    
    # tmp = pd.DataFrame({
    #     'x': [1e10],
    # })
    # sgap = -predict_model(best_model, tmp)['prediction_label'][0]
    # sgap = max(sgap, 0)
    
    # def g(x):
    #     tmp = pd.DataFrame({
    #         'x': [x],
    #     })
    #     return predict_model(best_model, tmp)['prediction_label'][0] + sgap * 1.05
        
    # for i in range(len(tt)):
    #     tmp = pd.DataFrame({
    #         'x': [tt[i]],
    #     })
    #     print(i, tt[i], vv[i], predict_model(best_model, tmp)['prediction_label'][0])

    # for i in range(10):
    #     tmp = pd.DataFrame({
    #         'x': [-10**i],
    #     })
    #     print(-10**i, predict_model(best_model, tmp)['prediction_label'][0])

    # for i in range(10):
    #     tmp = pd.DataFrame({
    #         'x': [10**i],
    #     })
    #     print(10**i, predict_model(best_model, tmp)['prediction_label'][0])


    try:
        stime = brentq(g, -1e10, 1e10)
    except:
        print("!!!!!!!!!!!!!!!!!!!!!!!!")
        print(g(-1e10), g(1e10))
        
    weee = stime
    assert stime < 1e10
    assert stime > 0
    stime = math.log(10 + stime) # 1 到 10.1
    k = 0.01
    return sgap, weee, 1 - math.exp(-k * sgap * 100), stime / 10.1 # 越小越好

def stime_c(time_list, val_list, lobj): # 收敛到收敛gap一定范围内的预估收敛时间
    if len(val_list) < 5:
        return -1, 0.5 # 数据太少，返回中等值
    vv, tt = val_list.copy(), time_list.copy()
    
    threshold = abs(vv[-1] - lobj) / vv[-1] if vv[-1] != 0 else 999999999  
    threshold /= len(time_list) * 5
    threshold = min(threshold, 1e-2)

    vv[0] = abs(vv[0] - lobj) / vv[0] if vv[0] != 0 else 999999999
    for i in range(1, len(vv)):
        gap_tmp = abs(vv[i] - lobj) / vv[i] if vv[i] != 0 else 999999999  
        vv[i] = min(gap_tmp, vv[i - 1] - threshold)

    for i in range(len(vv)):
        vv[i] = -vv[i]
    
    GAP = -1
    for i in range(len(tt) - 1):
        if abs(vv[i + 1] - vv[i]) / abs(vv[i + 1]) > 3:
           GAP = i 

    tt = tt[(GAP + 1):-1]
    vv = vv[(GAP + 1):-1]
    
    def func(x, b, c):
        return b * np.exp(c * x)
    popt, _ = curve_fit(func, tt, vv, maxfev=10000, bounds=([-1e16, -1e16], [0, 1e16]))

    def g(x):
        return func(x, *popt) + 1e-5
    
    stime = brentq(g, -1e10, 1e10)
    print(stime)
    wwwww = stime
    assert stime < 1e10
    assert stime > 0
    stime = math.log10(10 + stime) # 1 到 10.1
    k = 0.01
    return wwwww, stime / 10.1 # 越小越好

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
    early_gap = abs(early_val - lobj) / early_val if early_val != 0 else 999999999  
    k = 0.01
    return early_gap, 1 - math.exp(-k * early_gap * 100) # 越小越好

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
    medium_gap = abs(medium_val - lobj) / medium_val if medium_val != 0 else 999999999  
    k = 0.01
    return medium_gap, 1 - math.exp(-k * medium_gap * 100) # 越小越好

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
    if type == 'easy':
        threshold = 100
    elif type == 'medium':
        threshold = 600
    elif type == 'hard':
        threshold = 3500
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
    
    
def run():
    INSLIST = ["0", "1", "2"]
    for instance in instancelis:
        instance_name = os.path.basename(instance)
        weee = re.match(r".*([0-9]+)", instance_name).group(1)
        if weee not in INSLIST:
            continue
        for gr in grlis:
            for pre in prelis:
                for mod in modlis:
                    for sea in sealis:
                        if gr == "default":
                            if pre != "gurobi" and pre != "scip" and pre != "cplex":
                                continue
                            if mod != "default":
                                continue
                        else:
                            if pre == "gurobi" or pre == "scip" or pre == "cplex":
                                continue
                            if mod == "default":
                                continue
                        we = f"{gr}_{pre}_{mod}_{sea}_"
                        
                        subprocess.run(["python", "main.py", "--device", "cuda:2", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}", "--train_data_dir", f"{train_data_dir}",
                            "--graphencode", f"{gr}", "--predict", f"{pre}", "--modify", f"{mod}", "--search", f"{sea}", "--whole_time_limit", "100"])  # TODO: add error check  
                        
                        # instance_name = os.path.basename(instance)
                        # tmp = re.match(r"(.*)\.lp", instance_name)
                        # tmp = tmp.group(1)
    
                        # des = f'./logs/work/{args.taskname}/{we}/{tmp}_result.txt'
                        # if not os.path.exists(des):
                        #     continue
                        # with open(des, 'r') as f:
                        #     data = json.load(f)
                        # now_score[we] = calc(data, lobj)
        
        # LIS = []
        # for key, value in now_score.items():
        #     LIS.append(value)
        # SUM = calc_api(LIS)
        # for key, value in now_score.items():
        #     LIS.remove(value)
        #     sum = calc_api(LIS)
        #     scores[key] = scores.get(key, 0) + SUM - sum
        #     LIS.append(value)

    # for key, value in scores.items():
    #     print(key, value)    

def eval():
    result_list = {}
    for gr in grlis:
        for pre in prelis:
            for mod in modlis:
                for sea in sealis:
                    if gr == "default":
                        if pre != "gurobi" and pre != "scip" and pre != "cplex":
                            continue
                        if mod != "default":
                            continue
                    else:
                        if pre == "gurobi" or pre == "scip" or pre == "cplex":
                            continue
                        if mod == "default":
                            continue
 #                   if not (gr == "default" and pre == "gurobi" and mod == "default" and sea == "gurobi"):
#                        continue
                    
                    we = f"{gr}_{pre}_{mod}_{sea}_"
                    print(we)
                    scores = 0
                    cnt = 0
                    sum = 0
                    result_list_tmp = {}
#                    INSLIST = ["0", "1", "5", "6", "8"]
                    INSLIST = ["0", "1", "2"]
                    for instance in instancelis:
                        instance_name = os.path.basename(instance)
                        tmp = re.match(r"(.*)\.lp", instance_name)
                        tmp = tmp.group(1)
                        weee = re.match(r".*([0-9]+)", tmp).group(1)
                        if weee not in INSLIST:
                            continue
                        tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
                        des = f'./logs/work/{args.taskname}/{we}/{tmp_}/{tmp}_result.txt'
                        print(des)
                        if not os.path.exists(des):
                            continue
                        print(instance)
                        cnt += 1
                        lobj, type_ = work_gurobi(instance)
                        # if tmp[-1] != '6':
                            # continue
                        with open(des, 'r') as f:
                            data = json.load(f)
                        LISORI, LIS = calc(data, lobj, args.type)
                        data['indicatorsori'] = LISORI
                        data['indicators'] = LIS
                        ttttt = calc_api([LIS]) * 1e7
                        sum = sum + ttttt
                        data['score'] = ttttt
                        result_list_tmp[f'{tmp}'] = LIS + [ttttt]
                        with open(des, 'w') as f:
                            json.dump(data, f, indent=4)
                    if cnt != 3:
                        continue
                    result_list_tmp['score'] = sum / cnt
                    result_list[we] = result_list_tmp
    instance_name = os.path.basename(instance)
    tmp = re.match(r"(.*)_[0-9]+\.lp", instance_name)
    tmp = tmp.group(1)
    des = f'./logs/work/{args.taskname}/{tmp}_result.txt'
    result_list = dict(sorted(result_list.items(), key=lambda x: x[1]['score'], reverse=True))
    with open(des, 'w') as f:                
        json.dump(result_list, f, indent=4)
        
if args.eval:
    eval()
else :
    run()
        
        