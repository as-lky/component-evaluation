import os
import math
import numpy as np
import json
import subprocess
import re


taskname = "IS"

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
  
def calc(result_list, lobj, type):
    # 最终gap
    # TODO: 做数学推导
    time_list = [_[0] for _ in result_list]
    val_list = [_[1] for _ in result_list]

    if lobj > val_list[0]:
        lobj = max(lobj, val_list[-1])
    else:
        lobj = min(lobj, val_list[-1])

    if type == 'easy':
        if taskname == 'MT':
            threshold = 300
        else:
            threshold = 100
    elif type == 'medium':
        if taskname == 'IS':
            threshold = 600
        if taskname == 'MIKSC':
            threshold = 4000
        if taskname == 'SC':
            threshold = 600
        if taskname == 'MVC':
            threshold = 600
            
    elif type == 'hard':
        if taskname == 'IS':
            threshold = 3500
        if taskname == 'MIKSC':
            threshold = 8000
        if taskname == 'SC':
            threshold = 3500
        if taskname == 'MVC':
            threshold = 1800
    else:
        threshold = -1

    if time_list[-1] < threshold:
        time_list.append(threshold)
        val_list.append(val_list[-1])
    gapstartori, gapstart = gapstart_c(time_list, val_list, lobj)
    gapendori, gapend = gapend_c(time_list, val_list, lobj)
    nrori, nr = nr_c(time_list, val_list, lobj)
    yxtimeori, yxtime = yxtime_c(time_list, val_list, lobj)
    early_progressori, early_progress = early_progress_c(time_list, val_list, lobj)
    medium_progressori, medium_progress = medium_progress_c(time_list, val_list, lobj)
    area_under_curveori, area_under_curve = area_under_curve_c(time_list, val_list, lobj)
    
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


def func(data, task, type):
    taskname = task
    if taskname == "IS" and type == "hard":
        lobj = 252240
    if taskname == "IS" and type == "medium":
        lobj = 25361
    if taskname == "MIKSC" and type == "hard":
        lobj = 162
    if taskname == "MIKSC" and type == "medium":
        lobj = 16.52
    if taskname == "MVC" and type == "hard":
        lobj = 246449
    if taskname == "MVC" and type == "medium":
        lobj = 24575
    if taskname == "SC" and type == "hard":
        lobj = 2624
    if taskname == "SC" and type == "medium":
        lobj = 242
    
    if "val_list" in data:
        val_list = data['val_list']
        time_list = data['time_list']
        result_list = []
        for kk in range(len(val_list)):
            result_list.append((time_list[kk], val_list[kk]))
    else:
        result_list = data['result_list']
    LISORI, LIS = calc(result_list, lobj, type)
    return LISORI, LIS    

methodlis = ["./test/SCIP", "./test/gurobi", "./test/real", "./test/fake", "./test/light"]
typelis = ["hard", "medium"]
tasklis = ["IS", "MIKSC", "SC", "MVC"] 
for type in typelis:
    for task in tasklis:
        LIS = []
        Dic = {}
        for method in methodlis:
            we = os.path.join(method, f"{task}_{type}.txt")
            with open(we, "r") as f:
                data = json.load(f)
            ori, l = func(data, task, type)
            LIS.append(l)
            Dic[method] = l
        score = calc_api(LIS)
        for i, j in Dic.items():
            po = LIS.copy()
            po.remove(j)
            ss = (score - calc_api(po)) * 1e7
            out = f"{task} {type} {i} {ss}"
            
            print("!!!!!!!", score, ss)
#            with open('./test_tmp.txt', 'a') as f:
#                f.write(out)
#                f.write('\n')
  
# for i in os.listdir(we):
#     file = os.path.join(we, i)
#     with open(file, "r") as f:
#         data = json.load(f)

# #    nn = re.match(r".*_(.*)_(.*)_instance", i)
#     nn = re.match(r"(.*)_([hm])", i)
#     taskname = nn.group(1)
#     type = nn.group(2)
#     type = "hard" if type == "h" else "medium"

#     print("!!!!!!!", taskname, type)

#    result_list = data['result_list']
    

    # LISORI, LIS = calc(result_list, lobj, type)
    # if taskname == "SC" and type == "medium":
    #     print("===================")
    #     print(LISORI)
    #     print(LIS)
    #     print("===================")
        
    # with open('./test_tmp.txt', 'a') as f:
    #     f.write(file + " ")
    #     score = calc_api([LIS]) * 1e7
    #     f.write(str(score))
    #     f.write('\n')
