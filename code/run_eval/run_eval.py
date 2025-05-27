import os
import argparse
import re
import subprocess
import json
import math

parser = argparse.ArgumentParser(description="receive run or evaluate instruction")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "SC", "MKS"], help="taskname")
parser.add_argument("--instance_dir", type=str, required=True, help="the task instance input directory")
parser.add_argument("--type", type=str, help="easy medium hard")
parser.add_argument("--task", type=str, help="eval run task")
parser.add_argument("--eval", action="store_true", help="exec eval func")
parser.add_argument("--device", required=True, choices=["cpu", "cuda", "cuda:2", "cuda:1", "cuda:3"], help="cpu or cuda")

args = parser.parse_args()
instance_dir = args.instance_dir
instancelis = [os.path.join(instance_dir, file) for file in os.listdir(instance_dir)] # 30 instances

# get the instance's referring objective value and optimizing direction
# through running gurobi on the instance for enough long time
def work_gurobi(instance):
    # instance is the instance file path (*.lp)
    instance_name = os.path.basename(instance)
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
   
    if not os.path.exists(f'../logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'):
        subprocess.run(["python", "../main_streamline/main.py", "--device", "cuda", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}",
        "--graphencode", "test", "--predict", "gurobi", "--modify", "default", "--search", "gurobi", "--whole_time_limit", "3000"])    
    
    des = f'../logs/work/{args.taskname}/test_gurobi_default_gurobi_/{tmp_}/{tmp}_result.txt'
    with open(des, 'r') as f:
        data = json.load(f)
    
    if data['type'] == -1: # maximize problem
        return data['obj'] * (1 + data['gap'] / 100), -1
    else:
        return data['obj'] * (1 - data['gap'] / 100), 1
    
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

# get the running algorithms' components lists according to the particular task
grlis = ["bi", "bir", "tri", "trir", "default"]
prelis = ["cplex", "gcn", "gurobi", "scip", "gat"]
modlis = ["sr", "nr", "np", "default"]
sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP", "scip"]

if args.task == "cplex":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["cplex"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP", "scip"]
if args.task == "birgat":
    grlis = ["bir"]
    prelis = ["gat"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP", "scip"]
if args.task == "bigat":
    grlis = ["bi"]
    prelis = ["gat"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["gurobi", "LIH", "MIH", "LNS", "NALNS", "ACP", "scip"]
if args.task == "gurobisearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["gurobi"]
if args.task == "LIHsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["LIH"]
if args.task == "MIHsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["MIH"]
if args.task == "LNSsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["LNS"]
if args.task == "NALNSsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["NALNS"]
if args.task == "ACPsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["ACP"]
if args.task == "SCIPsearch":
    grlis = ["bi", "bir", "tri", "trir", "default"]
    prelis = ["gcn", "gurobi"]
    modlis = ["sr", "nr", "default", "np"]
    sealis = ["scip"]


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
    
# run the streamline main function to execute the task
# get some algorithm's performance on some instances
def run():
    # run 3 instances for each algorithm
    INSLIST = ["0", "1", "2"]
    for instance in instancelis:
        instance_name = os.path.basename(instance)
        weee = re.match(r".*([0-9]+)", instance_name).group(1)
        if weee not in INSLIST:
            continue
        # combine the components to form the algorithm and run the streamline main function
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
                        
                        subprocess.run(["python", "../main_steamline/main.py", "--device", f"{args.device}", "--taskname", f"{args.taskname}", "--instance_path", f"{instance}", 
                            "--graphencode", f"{gr}", "--predict", f"{pre}", "--modify", f"{mod}", "--search", f"{sea}", "--whole_time_limit", "100"]) 

# calculate the indicators of each algorithm's performance on each instance
def eval():
    # combine the components to form the algorithm name and get the result
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
                    
                    we = f"{gr}_{pre}_{mod}_{sea}_"
                    cnt = 0
                    sum = 0
                    result_list_tmp = {}

                    INSLIST = ["0", "1", "2"] 
                    for instance in instancelis:
                        instance_name = os.path.basename(instance)
                        tmp = re.match(r"(.*)\.lp", instance_name)
                        tmp = tmp.group(1)
                        weee = re.match(r".*([0-9]+)", tmp).group(1)
                        if weee not in INSLIST:
                            continue
                        tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
                        
                        # read the result file of the algorithm on the instance
                        des = f'../logs/work/{args.taskname}/{we}/{tmp_}/{tmp}_result.txt'
                        if not os.path.exists(des):
                            continue
                        cnt += 1
                        
                        # run the test-gurobi to get the dual bound of the instance 
                        lobj, type_ = work_gurobi(instance)

                        # read the result file of the algorithm on the instance and calculate the indicators
                        # and update the result data
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
                    result_list_tmp['score'] = sum / cnt if cnt != 0 else 0
                    result_list[we] = result_list_tmp

    # sort the result list by score and write to the output file
    instance_name = os.path.basename(instance)
    tmp = re.match(r"(.*)_[0-9]+\.lp", instance_name)
    tmp = tmp.group(1)
    des = f'../logs/work/{args.taskname}/{tmp}_result.txt'
    result_list = dict(sorted(result_list.items(), key=lambda x: x[1]['score'], reverse=True))
    with open(des, 'w') as f:                
        json.dump(result_list, f, indent=4)
        
if args.eval:
    eval()
else :
    run()        