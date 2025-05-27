import os
import argparse
import re
import subprocess
import json
import math

parser = argparse.ArgumentParser(description="receive evaluate instruction")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "SC", "MKS"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--train_data_dir", type=str, required=True, help="the train instances input folder")
parser.add_argument("--type", type=str, help="easy medium hard")
parser.add_argument("--task", type=str, help="eval run task")
parser.add_argument("--eval", action="store_true", help="exec eval func")
parser.add_argument("--device", required=True, choices=["cpu", "cuda", "cuda:2", "cuda:1", "cuda:3"], help="cpu or cuda")

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
    modlis = ["sr", "nr",  "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
elif args.task == "bigat":
    grlis = ["bi"]
    prelis = ["gat"]
    modlis = ["sr", "nr",  "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
elif args.task == "gurobisearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr",  "default"]
    sealis = ["gurobi"]
elif args.task == "LIHsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr",  "default"]
    sealis = ["LIH"]
elif args.task == "MIHsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr",  "default"]
    sealis = ["MIH"]
elif args.task == "LNSsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default"]
    sealis = ["LNS"]
elif args.task == "NALNSsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr",  "default"]
    sealis = ["NALNS"]
elif args.task == "ACPsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr",  "default"]
    sealis = ["ACP"]
elif args.task == "SCIPsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr",  "default"]
    sealis = ["scip"]
elif args.task == "cplex":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["cplex"]
    modlis = ["sr", "nr", "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP"]
else :
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["cplex", "gcn", "gurobi", "scip", "gat"]
    modlis = ["sr", "nr", "np", "default"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP", "scip"]


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
    if t >= 0.99:
        t = 0.99
    return gap, t # 越小越好

def gapend_c(time_list, val_list, lobj): # 最终gap
    val = val_list[-1]
    gap = abs(val - lobj) / lobj if lobj != 0 else 999999999  
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
        if args.taskname == 'MT':
            threshold = 300
        else:
            threshold = 100
    elif type == 'medium':
        if args.taskname == 'IS':
            threshold = 600
        if args.taskname == 'MIKS':
            threshold = 4000
        if args.taskname == 'SC':
            threshold = 600
        if args.taskname == 'MVC':
            threshold = 600
            
    elif type == 'hard':
        if args.taskname == 'IS':
            threshold = 3500
        if args.taskname == 'MIKS':
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
        nrori,
        yxtimeori,
        early_progressori,
        medium_progressori,
        area_under_curveori,
    ]
    
    ll1 = [ gapstart,
            gapend,
            nr,
            yxtime,
            early_progress,
            medium_progress,
            area_under_curve,
          ]
    return ll0, ll1


    # TODO:加入更多指标
    # 几个收敛指标?
    # 稳定性指标？
    # 加入大模型的
    ...
    
    
def run():
#    INSLIST = ["0", "1", "2"]
    INSLIST = ["1"]
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
                        
                        subprocess.run(["python", "main.py", "--device", f"{args.device}", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}", "--train_data_dir", f"{train_data_dir}",
                            "--graphencode", f"{gr}", "--predict", f"{pre}", "--modify", f"{mod}", "--search", f"{sea}", "--whole_time_limit", "60"])  # TODO: add error check  
                        
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
#                    INSLIST = ["0", "1", "2"]
                    INSLIST = ["1"]
 
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
                        ttttt = calc_api([LIS]) * 1e7 if len(LIS) >= 3 else (-LIS[0] if len(LIS) == 1 else -LIS[0] * LIS[1])
                        sum = sum + ttttt
                        data['score'] = ttttt
                        result_list_tmp[f'{tmp}'] = LIS + [ttttt]
                        with open(des, 'w') as f:
                            json.dump(data, f, indent=4)
#                    if cnt != 3:
#                        continue
                    result_list_tmp['score'] = sum / cnt if cnt != 0 else 0
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
        
        