import optuna
import os
import subprocess
import math
import re
import json
import argparse

parser = argparse.ArgumentParser(description="receive optimize instruction")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "SC", "MIKS"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
#parser.add_argument("--train_data_dir", type=str, required=True, help="the train instances input folder")
args = parser.parse_args()


def gapstart_c(time_list, val_list, lobj): # 初始解gap 
    val = val_list[0]
    gap = abs(val - lobj) / val if val != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * gap * 100) 
    if t >= 0.95:
        t = 0.95
    return gap, t # 越小越好

def gapend_c(time_list, val_list, lobj): # 最终gap
    val = val_list[-1]
    gap = abs(val - lobj) / val if val != 0 else 999999999  
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

def yxtime_c(time_list, val_list, lobj): # 第一个有效解和特优解的时间
    cnt = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.05:
            break
        cnt += 1
    ans1 = time_list[cnt] if cnt < len(time_list) else 99999990
    ans1g = math.log10(10 + ans1) # 1 到 8
    ans1g = ans1g / 20
    cnt = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.001:
            break
        cnt += 1
    ans2 = time_list[cnt] if cnt < len(time_list) else 99999990
    ans2g = math.log10(10 + ans2) # 1 到 8
    ans2g = ans2g / 20
    return ans1 + ans2, ans1g + ans2g # 越小越好


def calc(data, lobj):
    result_list = data['result_list']
    time_list = [_[0] for _ in result_list]
    val_list = [_[1] for _ in result_list]
    
    gapstartori, gapstart = gapstart_c(time_list, val_list, lobj)
    gapendori, gapend = gapend_c(time_list, val_list, lobj)
    irori, ir = ir_c(time_list, val_list, lobj)
    nrori, nr = nr_c(time_list, val_list, lobj)
    yxtimeori, yxtime = yxtime_c(time_list, val_list, lobj)
    
    ll0 = [
        gapstartori,
        gapendori,
        irori,
        nrori,
        yxtimeori,
    ]
    
    ll1 = [ gapstart,
            gapend,
            ir,
            nr,
            yxtime,
          ]
    return ll0, ll1

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


def work_gurobi(instance):
    instance_name = os.path.basename(instance)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
   
    if not os.path.exists(f'./logs/work/{args.taskname}/default_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'):
        subprocess.run(["python", "main.py", "--device", "cuda", "--args.taskname", f"{args.taskname}", "--instance_path", f"{instance}",
        "--graphencode", "default", "--predict", "gurobi", "--predict_time_limit", "30", "--modify", "default", "--search", "gurobi", "--search_time_limit", "30"])    
    
    des = f'./logs/work/{args.taskname}/default_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # 最大化
        return data['obj'] * (1 + data['gap'] / 100), -1
    else:
        return data['obj'] * (1 - data['gap'] / 100), 1



NUM_GPUS = 4

lobj, _ = work_gurobi(args.instance_path)

def objective(trial):
#    gpu_id = trial.number % NUM_GPUS
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    block = trial.suggest_int('block', 1, 5)
    ratio = trial.suggest_float('max_turn_ratio', 0.1, 0.5)
    exec = ['python', 'main.py', '--device', 'cuda:2', '--taskname', 'SC', '--instance_path', './Dataset/SC_medium_instance/SC_medium_instance/LP/SC_medium_instance_0.lp', 
            '--graphencode', 'default', '--predict', 'gurobi', '--predict_time_limit', '10', '--modify', 'default', '--search', 'ACP', '--search_time_limit', '1000']
    exec += ['--search_ACP_block', str(block), '--search_ACP_max_turn_ratio', str(ratio)]
    subprocess.run(exec)
    
    instance_name = os.path.basename(args.instance_path)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)


    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
    we = f"default_gurobi_default_ACP_{block}_{ratio}_"
    des = f'./logs/work/{args.taskname}/{we}/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    LISORI, LIS = calc(data, lobj)
    with open('./tmp.txt', 'a') as f:
        f.write(f"{block} {ratio} {LISORI} {LIS}\n")
    return calc_api([LIS]) * 1e7


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, n_jobs=4)  # 并行4个worker（=4块GPU）