from grapheme import endswith
from sympy import Sum
import numpy as np
import torch
import os
import argparse
import time
import re
import subprocess
import json
import shutil
import math
from scipy.optimize import brentq
from pycaret.regression import *

# python eval.py --taskname IS --instance_path ./Dataset/IS_easy_instance/IS_easy_instance/LP --train_data_dir ./Dataset/IS_easy_instance/IS_easy_instance/

parser = argparse.ArgumentParser(description="receive evaluate instruction")
parser.add_argument("--taskname", required=True, choices=["IP", "IS", "WA", "CA"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--train_data_dir", type=str, required=True, help="the train instances input folder")
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
   
    if not os.path.exists(f'./logs/work/{args.taskname}/default_gurobi_default_gurobi_/{tmp}_result.txt'):
        subprocess.run(["python", "main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
        "--graphencode", "default", "--predict", "gurobi", "--predict_time_limit", "60", "--modify", "default", "--search", "gurobi"])    
    
    des = f'./logs/work/{args.taskname}/default_gurobi_default_gurobi_/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # 最大化
        return data['obj'] * (1 + data['gap'] / 100), -1
    else:
        return data['obj'] * (1 - data['gap'] / 100), 1
    
instance_path = args.instance_path
train_data_dir = args.train_data_dir

grlis = ["bi", "bir", "tri", "trir", "default"]
prelis = ["gcn", "gurobi", "scip"]
#prelis = ["gat"]
modlis = ["sr", "nr", "np", "default"]
sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
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
    gap = abs(val - lobj) / val if val != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * gap * 100) 
    if t >= 0.95:
        t = 0.95
    return t # 越小越好

def gapend_c(time_list, val_list, lobj): # 最终gap
    val = val_list[-1]
    gap = abs(val - lobj) / val if val != 0 else 999999999  
    k = 0.01
    return 1 - math.exp(-k * gap * 100) # 越小越好


def ir_c(time_list, val_list, lobj): # 改进比率
    ir = abs(val_list[-1] - val_list[0]) / val_list[0] if val_list[0] != 0 else 99990
    if ir > 99990:
        ir = 99990
    # 差距可能很大！
    a = math.log(ir + 10) # 1 到 5
    a = 1 - a / 5
    return a # 越小越好

def nr_c(time_list, val_list, lobj): # 求解的有效率
    num = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.001:
            num += 1
    nr = num / len(val_list) 
    if num == 0: 
        nr = 0.5 / len(val_list)
    return 1 - nr # 越小越好

def sgap_stime_c(time_list, val_list, lobj): # 预估收敛gap & 收敛到预估收敛gap一定范围内的预估收敛时间
    if len(val_list) < 2:
        return 0.5 # 数据太少，返回中等值
    type = -1 if lobj < val_list[0] else 1
    vv, tt = val_list.copy(), time_list.copy()
    
    threshold = abs(vv[-1] - lobj) / vv[-1] if vv[-1] != 0 else 999999999  
    threshold /= len(time_list) * 5
    threshold = min(threshold, 1e-8)

    for i in range(1, len(vv)):
        gap_tmp = abs(vv[i] - lobj) / vv[i] if vv[i] != 0 else 999999999  
        vv[i] = min(gap_tmp, vv[i - 1] - threshold)

    for i in range(len(vv)):
        vv[i] = -vv[i]
        
    import pandas as pd
    data = pd.DataFrame({'x': time_list, 'y': vv})
    reg = setup(data, target='y', session_id=123)
    best_model = compare_models()
    
    tmp = pd.DataFrame({
        'x': [1e10],
    })
    print(predict_model(best_model, tmp))
    sgap = -predict_model(best_model, tmp)['prediction_label'][0]
    sgap = max(sgap, 0)
    
    def g(x):
        tmp = pd.DataFrame({
            'x': [x],
        })
        return predict_model(best_model, tmp)['prediction_label'][0] + sgap * 1.05
        
    stime = brentq(g, -1e10, 1e10)
    assert stime < 1e10
    assert stime > 0
    stime = math.log(10 + stime) # 1 到 10.1
    k = 0.01
    return 1 - math.exp(-k * sgap * 100), stime / 10.1 # 越小越好

def imprate_c(time_list, val_list, lobj):
    if len(val_list) < 2:
        return 0.5  # 数据太少，返回中等值
    total_time = time_list[-1] - time_list[0]
    improve = abs(val_list[0] - val_list[-1])
    rate = improve / total_time
    norm_rate = 1 / (rate + 1e-5)  # 改进越快值越小
    norm_rate = min(norm_rate, 10)
    return norm_rate / 10  # 越小越好

def stability_c(time_list, val_list, lobj):
    diffs = [abs(val_list[i+1] - val_list[i]) for i in range(len(val_list)-1)]
    if not diffs:
        return 0
    ss = 0
    for diff in diffs:
        ss += diff
    avg_diff = ss / len(diffs)
    norm_diff = min(avg_diff / (abs(val_list[0]) + 1e-6), 1)
    return norm_diff  # 越小越好

def earlygap_c(time_list, val_list, lobj):
    total_time = time_list[-1]
    threshold_time = total_time * 0.1
    early_vals = [val_list[i] for i in range(len(time_list)) if time_list[i] <= threshold_time]
    if not early_vals:
        early_vals = [val_list[0]]
    gaps = [abs(v - lobj) / v if v != 0 else 1 for v in early_vals]
    ss = 0
    for gap in gaps:
        ss += gap
    mean_gap = ss / len(gaps)
    return min(mean_gap, 1)  # 越小越好

def trend_c(time_list, val_list, lobj):
    if len(val_list) < 3:
        return 0.5  # 数据太少，返回中等值

    # 统计每一步是否改进了
    improvements = 0
    for i in range(1, len(val_list)):
        if val_list[i] < val_list[i - 1]:
            improvements += 1
    trend_ratio = improvements / (len(val_list) - 1)

    # 越接近持续改进，trend_ratio 越接近 1，越好，取反归一化
    return 1 - trend_ratio  # 越小越好

#TODO:不严格单调的噢!
#TODO:求解的稳定性指标?

def calc(data, lobj):
    # 最终gap
    # TODO: 做数学推导
    result_list = data['result_list']
    time_list = [_[0] for _ in result_list]
    val_list = [_[1] for _ in result_list]
    
    gapstart = gapstart_c(time_list, val_list, lobj)
    gapend = gapend_c(time_list, val_list, lobj)
    ir = ir_c(time_list, val_list, lobj)
    nr = nr_c(time_list, val_list, lobj)
    sgap, stime = sgap_stime_c(time_list, val_list, lobj)
    
    ll1 = [ gapstart,
            gapend,
            ir,
            nr,
            stime,
            sgap,
#          imprate_c(time_list, val_list, lobj),
#          stability_c(time_list, val_list, lobj),
#          earlygap_c(time_list, val_list, lobj),
#          trend_c(time_list, val_list, lobj),
          # TODO: 使用大模型指标
          ]
    ll2 = [0.05, 0.4, 0.15, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05]
 #   for i in range(len(ll1)):
#        ll1[i] = 1 - (1 - ll1[i]) * ll2[i] / 0.4
 #       print(ll1[i])
 #       ll1[i] /= 2
    print(ll1)
    return ll1


    # TODO:加入更多指标
    # 几个收敛指标?
    # 稳定性指标？
    # 加入大模型的
    ...



result_list = []
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
                if not (we == "bi_gcn_sr_LIH_" or we == "default_gurobi_default_gurobi_"):
                    continue
                scores = 0
                cnt = 0
                sum = 0
                print("________________________________________")
                print(we)
                for instance in instancelis:
                    print(instance)
                    if cnt >= 5:
                        break    
                    instance_name = os.path.basename(instance)
                    tmp = re.match(r"(.*)\.lp", instance_name)
                    tmp = tmp.group(1)
                    des = f'./logs/work/{args.taskname}/{we}/{tmp}_result.txt'
                    if not os.path.exists(des):
                        continue
                    cnt += 1
                    lobj, type_ = work_gurobi(instance)

                    with open(des, 'r') as f:
                        data = json.load(f)
                    LIS = calc(data, lobj)
                    sum = sum + calc_api([LIS])
                
                if cnt != 5:
                    continue
                
                if cnt == 0:
                    result_list.append((we, sum))
                else:
                    result_list.append((we, sum / cnt))

# instance_name = os.path.basename(instance)
# tmp = re.match(r"(.*)_[0-9]+\.lp", instance_name)
# tmp = tmp.group(1)
# des = f'./logs/work/{args.taskname}/{tmp}_result.txt'
# with open(des, 'w') as f:                
#     json.dump(result_list, f, indent=4)
        