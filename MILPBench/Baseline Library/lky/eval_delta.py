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
args = parser.parse_args()

instance_path = args.instance_path

instancelis = [os.path.join(instance_path, file) for file in os.listdir(instance_path)] # 30 instances
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


instance = instancelis[0]
instance_name = os.path.basename(instance)
tmp = re.match(r"(.*)_[0-9]+\.lp", instance_name)
tmp = tmp.group(1)
des = f'./logs/work/{args.taskname}/{tmp}_result.txt'
with open(des, 'r') as f:
    result_list = json.load(f)

def domi(a, b):
    # a dominates b
    l = len(a)
    for i in range(l):
        if a[i] > b[i]:
            return False
    return True 

#INSLIST = ["IS_easy_instance_6", "IS_easy_instance_5", "IS_easy_instance_8", "IS_easy_instance_1", "IS_easy_instance_0"]
INSLIST = ["SC_easy_instance_0", "SC_easy_instance_1", "SC_easy_instance_2"]
#INSLIST = ["IS_easy_instance_0", "IS_easy_instance_1", "IS_easy_instance_2"]
#INSLIST = ["MVC_easy_instance_0", "MVC_easy_instance_1", "MVC_easy_instance_2"]
#INSLIST = ["MIKS_easy_instance_0", "MIKS_easy_instance_1", "MIKS_easy_instance_2"]

SCORES = {}
for a, b in result_list.items():
    SCORES[a] = 0

for instance in INSLIST:
    tmp = []
    for i, value in result_list.items():
        flag = 0
        for j, value2 in result_list.items():
            if i == j:
                continue
            if domi(value2[instance][:-1], value[instance][:-1]):
                flag = 1
                break
        if flag == 0:
            tmp.append(i)
    print("WIN : ", tmp)
    if len(tmp) == 1:
        SCORES[tmp[0]] += calc_api([result_list[tmp[0]][instance][:-1]]) * 1e7
    else:
        er = []
        for i in tmp:
            er.append(result_list[i][instance][:-1])
        score = calc_api(er)
        for i in tmp:
            po = er.copy()
            po.remove(result_list[i][instance][:-1])
            SCORES[i] += (score - calc_api(po)) * 1e7
    wee = "bir_gcn_sr_LNS_"
    if wee in SCORES:
        with open("./eval_delta_log.txt", 'a') as f:
            f.write(f"{wee} {SCORES[wee]}\n")
            
instance = instancelis[0]
instance_name = os.path.basename(instance)
tmp = re.match(r"(.*)_[0-9]+\.lp", instance_name)
tmp = tmp.group(1)
des = f'./logs/work/{args.taskname}/{tmp}_result_delta.txt'
SCORES = dict(sorted(SCORES.items(), key=lambda x: x[1], reverse=True))
with open(des, 'w') as f:                
    json.dump(SCORES, f, indent=4)
