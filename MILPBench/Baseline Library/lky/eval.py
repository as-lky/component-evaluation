from grapheme import endswith
import torch
import os
import argparse
import time
import re
import subprocess
import json
import shutil

parser = argparse.ArgumentParser(description="receive evaluate instruction")
parser.add_argument("--taskname", required=True, choices=["IP", "IS", "WA", "CA"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
args = parser.parse_args()

def c(a):
    tmp = os.path.basename(a)
    tmp = re.match(r".*_([0-9]+)", tmp)
    tmp = tmp.group(1)
    return int(tmp) <= 9


def work_gurobi(instance):
    instance_name = os.path.basename(instance)

    subprocess.run(["python", "main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
    "--graphencode", "default", "--predict", "gurobi", "--predict_time_limit", "30", "--modify", "default", "--search", "gurobi"])    
    des = f'./logs/work/{args.taskname}/{instance_name}/default_gurobi_dafault_gurobi_/result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # 最大化
        return data['obj'] * (1 + data['gap'] / 100)
    else:
        return data['obj'] * (1 - data['gap'] / 100)
    
instance_path = args.instance_path

grlis = ["bi", "tri", "bir", "trir", "default"]
prelis = ["l2bs", "gtran", "gat", "gcn", "gurobi", "scip", "cplex"]
modlis = ["sr", "nr", "np", "default"]
sealis = ["gurobi", "scip", "LIH", "MIH", "LNS", "NALNS", "ACP"]
instancelis = [file for file in os.listdir(instance_path) if c(file)] # 10 instances

score_dic = {}

def calc(now_score):
    if os.path.exists('calc/hbda/build/nonincremental/tmp.txt'):
        os.remove('calc/hbda/build/nonincremental/tmp.txt')
    SUM = 0
    with open('calc/hbda/build/nonincremental/tmp.txt', 'w') as f:
        for _ in now_score.keys():
            for we in range(3):
                for __ in _:
                    f.write(f"{__} ")
            f.write("\n")
        subprocess.run(['calc/hbda/build/nonincremental/nonincremental', '-O', 'calc/hbda/build/nonincremental/tmp.txt'])
        
    return 0

for instance in instancelis:
    lobj = work_gurobi(instance)
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
                    
                    subprocess.run(["python", "main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
                        "--graphencode", f"{gr}", "--predict", f"{pre}", "--predict_time_limit", "15", "--modify", f"{mod}", "--search", f"{sea}", "--search_time_limit", "15"])    
                    instance_name = os.path.basename(instance)
                    des = f'./logs/work/{args.taskname}/{instance}/{we}/result.txt'
                    with open(des, 'r') as f:
                        data = json.load(f)
                    now_score[we] = [abs(data['obj'] - lobj) / data['obj'] * 100]
    calc(now_score)        
    
    