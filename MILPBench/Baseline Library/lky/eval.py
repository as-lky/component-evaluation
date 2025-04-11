from grapheme import endswith
import torch
import os
import argparse
import time
import re
import subprocess
import json
import shutil
import math
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar


parser = argparse.ArgumentParser(description="receive evaluate instruction")
parser.add_argument("--taskname", required=True, choices=["IP", "IS", "WA", "CA"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--train_data_dir", type=str, required=True, help="the train instances input folder")
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
   

    subprocess.run(["python", "main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
    "--graphencode", "default", "--predict", "gurobi", "--predict_time_limit", "10", "--modify", "default", "--search", "gurobi"])    
    des = f'./logs/work/{args.taskname}/default_gurobi_default_gurobi_/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # 最大化
        return data['obj'] * (1 + data['gap'] / 100), -1
    else:
        return data['obj'] * (1 - data['gap'] / 100), 1
    
instance_path = args.instance_path
train_data_dir = args.train_data_dir

grlis = ["bi", "bir", "default"]
prelis = ["gcn", "gurobi", "scip", "cplex"]
modlis = ["sr", "nr", "np", "default"]
sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
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
    subprocess.run(['./calc/hbda/build/nonincremental/nonincremental', '-O', 'calc/hbda/build/nonincremental/tmp.txt'])    
    return 0

def gapstart_c(time_list, val_list, lobj): # 初始解gap
    val = val_list[0]
    gap = abs(val - lobj) / val 
    return min(gap / 1000, 1) # 越小越好

def gapend_c(time_list, val_list, lobj): # 最终gap
    val = val_list[-1]
    gap = abs(val - lobj) / val 
    return min(gap / 10, 1) # 越小越好

def ir_c(time_list, val_list, lobj): # 改进比率
    ir = abs(val_list[-1] - val_list[0]) / val_list[0]
    if ir < 0.01:
        ir = 0.01
    if ir > 100000:
        ir = 100000
    # 差距可能很大！
    a = math.log(ir) # -2 到 5
    a = 1 - (a + 2) / 7
    return a # 越小越好

def nr_c(time_list, val_list, lobj): # 求解的有效率
    num = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.001:
            num += 1
    nr = num / len(val_list) 
    return 1 - nr # 越小越好

def stime_c(time_list, val_list, lobj): # 求解的预估收敛时间
    pchip = PchipInterpolator(time_list, val_list)    
    def func(x):
        return pchip(x) - lobj
    stime = root_scalar(func, bracket=[0.1, 1000000])
    if stime.converged:
        stime = math.log(stime.root) # -1 到 6
        return (stime + 1) / 7 # 越小越好
    else:
        return 1 # 越小越好

#TODO:check求解的稳定性

def calc(data, lobj):
    # 最终gap
    # TODO: 做数学推导
    result_list = data['result_list']
    time_list = [_[0] for _ in result_list]
    val_list = [_[1] for _ in result_list]
    
    ll1 = [gapstart_c(time_list, val_list, lobj), 
          gapend_c(time_list, val_list, lobj), 
          ir_c(time_list, val_list, lobj),
          nr_c(time_list, val_list, lobj),
          stime_c(time_list, val_list, lobj)]
    ll2 = [0.1, 0.4, 0.2, 0.1, 0.2]
    for i in range(len(ll1)):
        ll1[i] *= ll2[i]
        ll1[i] /= 0.4  
    return ll1

    # TODO:加入更多指标
    # 几个收敛指标?
    # 稳定性指标？
    # 加入大模型的
    ...
    
scores = {}

for instance in instancelis:
    lobj, type_ = work_gurobi(instance)
    now_score = {}
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
                    
                    subprocess.run(["python", "main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}", "--train_data_dir", f"{train_data_dir}",
                        "--graphencode", f"{gr}", "--predict", f"{pre}", "--predict_time_limit", "15", "--modify", f"{mod}", "--search", f"{sea}", "--search_time_limit", "15"])  # TODO: add error check  
                    
                    instance_name = os.path.basename(instance)
                    tmp = re.match(r"(.*)\.lp", instance_name)
                    tmp = tmp.group(1)
   
                    des = f'./logs/work/{args.taskname}/{we}/{tmp}_result.txt'
                    with open(des, 'r') as f:
                        data = json.load(f)
                    now_score[we] = calc(data, lobj)
    
    LIS = []
    for key, value in now_score.items():
        LIS.append(value)
    SUM = calc_api(LIS)
    for key, value in now_score.items():
        LIS.remove(value)
        sum = calc_api(LIS)
        scores[we] = scores.get(we, 0) + SUM - sum
        LIS.append(value)

for key, value in scores.items():
    print(key, value)    
    