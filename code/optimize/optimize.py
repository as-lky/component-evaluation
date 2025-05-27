import optuna
import os
import subprocess
import math
import re
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="receive optimize instruction")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "SC", "MKS"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--type", type=str, required=True, choices=["easy", "medium", "hard"], help="the task type")

args = parser.parse_args()


# get the hypervolume score of some algorithms' perform vectors on a constant instance referring to (1, 1, ..., 1)
def calc_api(lis):
    # lis is a list in which each element is a list, corresponding to a point vector meaning an algorithm's performance on an instance
    # each dim of the vector is a float number in [0, 1)
    if os.path.exists('../calc/hbda/build/nonincremental/tmp.txt'):
        os.remove('../calc/hbda/build/nonincremental/tmp.txt')
    SUM = 0
    # write the list input into a tmp file
    with open('../calc/hbda/build/nonincremental/tmp.txt', 'w') as f:
        for i in lis:
            for j in i:
                w = round(j, 8)
                f.write(f"{w} ")
            f.write("\n")
    if os.path.exists('../calc/hbda/build/nonincremental/result.txt'):
        os.remove('../calc/hbda/build/nonincremental/result.txt')

    # run the hypervolume calculation program
    subprocess.run(['../calc/hbda/build/nonincremental/nonincremental', '-O', '../calc/hbda/build/nonincremental/tmp.txt', '-S', '../calc/hbda/build/nonincremental/result.txt'])

    # read the result from the output file
    with open('../calc/hbda/build/nonincremental/result.txt', 'r') as f:
        lines = f.readlines()
        line_result = lines[1].strip()
        result = line_result.split()
        SUM = float(result[-1])
    return SUM

# Indicator: the initial gap after predict and repair 
def gapstart_c(time_list, val_list, lobj):
    # time_list, val_list are both a list of floats, (time_list[i], val_list[i]) means the algorithm get val_list[i] at time_list[i]
    # lobj is a float, meaning dual bound of the instance 
    val = val_list[0]
    gap = abs(val - lobj) / lobj if lobj != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * gap * 100) 
    if t >= 0.99:
        t = 0.99
    return gap, t # the original gap and the projection of the gap to [0,1), the smaller the better

# Indicator: the final gap after search 
def gapend_c(time_list, val_list, lobj): 
    # time_list, val_list are both a list of floats, (time_list[i], val_list[i]) means the algorithm get val_list[i] at time_list[i]
    # lobj is a float, meaning dual bound of the instance 
    val = val_list[-1]
    gap = abs(val - lobj) / lobj if lobj != 0 else 999999999  
    k = 0.01
    t = 1 - math.exp(-k * gap * 100) 
    if t >= 0.99:
        t = 0.99
    return gap, t # the original gap and the projection of the gap to [0,1), the smaller the better

# Indicator: the efficiency rate 
def nr_c(time_list, val_list, lobj):
    # time_list, val_list are both a list of floats, (time_list[i], val_list[i]) means the algorithm get val_list[i] at time_list[i]
    # lobj is a float, meaning dual bound of the instance 
    num = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.1:
            num += 1
    nr = num / len(val_list) 
    if num == 0: 
        nr = 0.0001
    return num / len(val_list), 1 - nr # the original rate and the projection of the rate to [0,1), the smaller the better

# Indicator: time to valid and high-quality solutions
def yxtime_c(time_list, val_list, lobj): 
    # time_list, val_list are both a list of floats, (time_list[i], val_list[i]) means the algorithm get val_list[i] at time_list[i]
    # lobj is a float, meaning dual bound of the instance 
    cnt = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.1:
            break
        cnt += 1
    ans1 = time_list[cnt] if cnt < len(time_list) else 99999990
    ans1g = math.log10(10 + ans1) # 1 to 8
    ans1g = ans1g / 20
    cnt = 0
    for _ in val_list:
        if abs(_ - lobj) / lobj < 0.01:
            break
        cnt += 1
    ans2 = time_list[cnt] if cnt < len(time_list) else 99999990
    ans2g = math.log10(10 + ans2) # 1 to 8
    ans2g = ans2g / 20
    return ans1 + ans2, ans1g + ans2g # the original time sum and the projection of the time sum to [0,1), the smaller the better
  
# Indicator: solution gap at 20% of the time budget
def early_progress_c(time_list, val_list, lobj):
    # time_list, val_list are both a list of floats, (time_list[i], val_list[i]) means the algorithm get val_list[i] at time_list[i]
    # lobj is a float, meaning dual bound of the instance 
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
    return early_gap, t # the original gap and the projection of the gap to [0,1), the smaller the better

# Indicator: solution gap at 60% of the time budget
def medium_progress_c(time_list, val_list, lobj):
    # time_list, val_list are both a list of floats, (time_list[i], val_list[i]) means the algorithm get val_list[i] at time_list[i]
    # lobj is a float, meaning dual bound of the instance 
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
    return medium_gap, t # the original gap and the projection of the gap to [0,1), the smaller the better

# Indicator: time-integrated solution gap over the full horizon (area under the curve)
def area_under_curve_c(time_list, val_list, lobj):
    # time_list, val_list are both a list of floats, (time_list[i], val_list[i]) means the algorithm get val_list[i] at time_list[i]
    # lobj is a float, meaning dual bound of the instance 
    if len(time_list) < 2:
        return -1, 0.5 # return middle value when the list is too short
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
    log_auc = math.log10(10 + auc)  # 1 to 4
    return auc, (log_auc - 1) / 4  # the original integral value and the projection of the value to [0,1), the smaller the better


# get the indicators vector of an algorithm's performance through the solving list
def calc(data, lobj, type):
    # data is a dict, with key:
    # 'gap': the final gap of some algorithm
    # 'obj': the final objective value of some algorithm
    # 'type': the optimization direction of the instance, -1 for maximization, 1 for minimization
    # 'result_list': a list of tuples, each tuple is (time, value) meaning the algorithm get value at time
    
    # lobj is a float, meaning the dual bound of the instance
    # type is "easy", "medium" or "hard", meaning the problem scale
    result_list = data['result_list']
    time_list = [_[0] for _ in result_list]
    val_list = [_[1] for _ in result_list]

    # most time lobj will be better than the final gap of the algorithm
    if lobj > val_list[0]:
        lobj = max(lobj, val_list[-1])
    else:
        lobj = min(lobj, val_list[-1])

    # get the time budget threshold for the problem scale
    if type == 'easy':
        threshold = 100
    elif type == 'medium':
        if args.taskname == 'IS':
            threshold = 600
        if args.taskname == 'MKS':
            threshold = 4000
        if args.taskname == 'SC':
            threshold = 600
        if args.taskname == 'MVC':
            threshold = 600
    elif type == 'hard':
        if args.taskname == 'IS':
            threshold = 3500
        if args.taskname == 'MKS':
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

    # calculate the indicators
    gapstartori, gapstart = gapstart_c(time_list, val_list, lobj)
    gapendori, gapend = gapend_c(time_list, val_list, lobj)
    nrori, nr = nr_c(time_list, val_list, lobj)
    yxtimeori, yxtime = yxtime_c(time_list, val_list, lobj)
    early_progressori, early_progress = early_progress_c(time_list, val_list, lobj)
    medium_progressori, medium_progress = medium_progress_c(time_list, val_list, lobj)
    area_under_curveori, area_under_curve = area_under_curve_c(time_list, val_list, lobj)
    
    # the list of original indicators
    ll0 = [
        gapstartori,
        gapendori,
        nrori,
        yxtimeori,
        early_progressori,
        medium_progressori,
        area_under_curveori,
    ]
    
    # the list of projected indicators
    ll1 = [ gapstart,
            gapend,
            nr,
            yxtime,
            early_progress,
            medium_progress,
            area_under_curve,
          ]
    return ll0, ll1
    

# get the instance's referring objective value and optimizing direction
# through running gurobi on the instance for enough long time
def work_gurobi(instance):
    # instance is the instance file path (*.lp)
    instance_name = os.path.basename(instance)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
   
    if not os.path.exists(f'../logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'):
        # get running time according to the problem scale
        rt = 3000
        if args.type == 'medium':
            rt = 12000
        if args.type == 'hard':
            rt = 30000
        subprocess.run(["python", "../main_streamline/main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
        "--graphencode", "test", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", f"{rt}"])    
    
    des = f'../logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # maximize problem
        return data['obj'] * (1 + data['gap'] / 100), -1
    else:
        return data['obj'] * (1 - data['gap'] / 100), 1


lobj, _ = work_gurobi(args.instance_path)

# get the optimized objective under some hyperparameters set
# and using optuna to optimize the hyperparameters
def objective(trial):
    # using IS easy instance as an example
    type_ = args.type
    # bi gcn sr ACP algorithm running 600s
    exec = ['python', '../main_streamline/main.py', '--device', 'cuda:1', '--taskname', 'IS', '--instance_path', f'{args.insntace_path}', 
            '--graphencode', 'bi', '--predict', 'gcn', '--whole_time_limit', '600', '--modify', 'sr', '--search', 'ACP']

    block = trial.suggest_int('block', 2, 10)   
    ratio = trial.suggest_float('ratio', 0.1, 0.9)
    exec += ['--search_ACP_LNS_block', str(block), '--search_ACP_LNS_max_turn_ratio', str(ratio)]
 
    subprocess.run(exec)
    
    instance_name = os.path.basename(args.instance_path)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)

    we = f'bi_gcn_sr_ACP_{block}_{ratio}_'  # the algorithm sequence name (example)
    
    # read the result from the main streamline output path
    des = f'../logs/work/{args.taskname}/{we}/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)

    # calculate the indicators and hypervolume score
    LISORI, LIS = calc(data, lobj, type_)
    result___ = calc_api([LIS]) * 1e7
    return result___

# using postgresql to multiprocess the optimization
# should create study based on sql storage before loading study 
study = optuna.load_study(storage="postgresql://...", study_name="...")
study.optimize(objective, n_trials=5)
