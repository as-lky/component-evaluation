import os
import numpy as np
import pyscipopt as scp
import time
import random
import gurobipy as gp
import pickle
import re

from .help.LIH.help import greedy_one as greedy_one_LIH, split_problem as split_problem_LIH
from .help.MIH.help import greedy_one as greedy_one_MIH, split_problem as split_problem_MIH
from .help.NALNS.help import greedy_one as greedy_one_NALNS, split_problem as split_problem_NALNS
from .help.LNS.help import split_problem as split_problem_LNS
from .help.ACP.help import split_problem as split_problem_ACP
from .mod import Component, Modify2Search, Cansol2S

from pyscipopt import SCIP_PARAMSETTING, Eventhdlr, SCIP_EVENTTYPE
from typing import Type, cast
from gurobipy import GRB


class Search(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "scip":
            cls = SCIP
        elif component == "gurobi":
            cls = Gurobi
        elif component == "LIH":
            cls = LIH
        elif component == "MIH":
            cls = MIH
        elif component == "LNS":
            cls = LNS
        elif component == "NALNS":
            cls = NALNS
        elif component == "ACP":
            cls = ACP
        else:
            raise ValueError("Search component type is not defined")

        return super().__new__( cls )
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)

    def work(self, input: Modify2Search, result_list: list):...
        
class LIH(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        self.choose = kwargs.get("choose") or 0.5
        self.set_pa = kwargs.get("set_pa") or 0.3
        ... # tackle parameters

    def work(self, input: Cansol2S, result_list: list):
        self.begin()
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type = split_problem_LIH(self.instance)
        new_sol = []
        
        tmp = gp.read(self.instance)
        for var in tmp.getVars():
            new_sol.append(input.cansol[var.VarName])
            
        time_limit = self.time_limit
        new_site = []
        new_value = []
        new_constraint = np.zeros(m)
        new_constraint_type = np.zeros(m, int)
        for i in range(m):
            new_site.append(np.zeros(k[i], int))
            new_value.append(np.zeros(k[i]))
            for j in range(k[i]):
                new_site[i][j] = site[i][j]
                new_value[i][j] = value[i][j]
            new_constraint[i] = constraint[i]
            new_constraint_type[i] = constraint_type[i]
        
        new_coefficient = np.zeros(n)
        new_lower_bound = np.zeros(n)
        new_upper_bound = np.zeros(n)
        new_value_type = np.zeros(n, int)
        new_new_sol = np.zeros(n)
        for i in range(n):
            new_coefficient[i] = coefficient[i]
            new_lower_bound[i] = lower_bound[i]
            new_upper_bound[i] = upper_bound[i]
            if(value_type[i] == 'B'):
                new_value_type[i] = 0
            elif(value_type[i] == 'C'):
                new_value_type[i] = 1
            else:
                new_value_type[i] = 2
            new_new_sol[i] = new_sol[i]

        now_instance = (n, m, k, new_site, new_value, new_constraint, new_constraint_type, new_coefficient, obj_type, new_lower_bound, new_upper_bound, new_value_type, new_new_sol)
        
        now_sol, now_time, now_gap = greedy_one_LIH(now_instance, time_limit, self.choose, self.set_pa)

        for _ in range(len(now_sol)):
            result_list.append((result_list[0][0] + now_time[_], now_sol[_]))

        self.end()
        return 0, 0, 0 

 
class MIH(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        self.choose = kwargs.get("choose") or 0.5
        self.set_pa = kwargs.get("set_pa") or 0.3
 
        ... # tackle parameters

    def work(self, input: Cansol2S, result_list: list):
        self.begin()
        
        ns_ = []
        
        tmp = gp.read(self.instance)
        for var in tmp.getVars():
            ns_.append(input.cansol[var.VarName])
        
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type = split_problem_MIH(self.instance)
        time_limit = self.time_limit
        new_site = []
        new_value = []
        new_constraint = np.zeros(m)
        new_constraint_type = np.zeros(m, int)
        for i in range(m):
            new_site.append(np.zeros(k[i], int))
            new_value.append(np.zeros(k[i]))
            for j in range(k[i]):
                new_site[i][j] = site[i][j]
                new_value[i][j] = value[i][j]
            new_constraint[i] = constraint[i]
            new_constraint_type[i] = constraint_type[i]
        
        new_coefficient = np.zeros(n)
        new_lower_bound = np.zeros(n)
        new_upper_bound = np.zeros(n)
        new_value_type = np.zeros(n, int)
        new_new_sol = np.zeros(n)
        for i in range(n):
            new_coefficient[i] = coefficient[i]
            new_lower_bound[i] = lower_bound[i]
            new_upper_bound[i] = upper_bound[i]
            if(value_type[i] == 'B'):
                new_value_type[i] = 0
            elif(value_type[i] == 'C'):
                new_value_type[i] = 1
            else:
                new_value_type[i] = 2
            new_new_sol[i] = ns_[i]

        now_instance = (n, m, k, new_site, new_value, new_constraint, new_constraint_type, new_coefficient, obj_type, new_lower_bound, new_upper_bound, new_value_type, new_new_sol)
        now_sol, now_time, now_gap = greedy_one_MIH(now_instance, time_limit, self.choose, self.set_pa)

        for _ in range(len(now_sol)):
            result_list.append((result_list[0][0] + now_time[_], now_sol[_]))

        self.end()
        return 0, 0, 0 
    
class LNS(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        self.block = kwargs.get("block") or 4
        self.max_turn_ratio = kwargs.get("max_turn_ratio") or 0.01
        
        ... # tackle parameters

    def work(self, input: Cansol2S, result_list: list):
        self.begin()
        
        time_limit = self.time_limit
        max_turn_time = self.max_turn_ratio * self.time_limit
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num = split_problem_LNS(self.instance)
        # Build a variable dictionar
        num_to_value = {value : key for key, value in value_to_num.items()}
        #Set KK as the initial number of blocks, and PP as the selected number of blocks to optimize after dividing the constraints into KK blocks
        KK = self.block
        
        ns_ = []
        
        tmp = gp.read(self.instance)
        for var in tmp.getVars():
            ns_.append(input.cansol[var.VarName])
        
        ans, ansx = input.objval, ns_
        # Here the ans is the objective value, ansx is the sol of index(like ans in LIH MIH)
        print(f"Initial objective: {ans}")
                
        begin_time = time.time()
        GAP = input.gap
        
        now_sol, now_time = [], []
        
        while(time.time() - begin_time <= time_limit):
            print("KK = ", KK)
            #Randomly divide the decision variables into KK blocks
            color = np.zeros(n, int)
            for i in range(n):
                color[i] = random.randint(1, KK)
            #Enumerate each of the blocks
            for now_color in range(1, KK + 1):
                #Exit when reaching the time limit
                if(time.time() - begin_time > time_limit):
                    break
                #site_to_color[i]represents which decision variable is the i-th decision variable in this block
                #color_to_site[i]represents which decision variable is mapped to the i-th decision variable in this block
                #vertex_color_num represents the number of decision variables in this block currently
                site_to_color = np.zeros(n, int)
                color_to_site = np.zeros(n, int)
                vertex_color_num = 0

                #Define the model to solve
                model = gp.Model("LNS")
                #Define decision variables x[]
                x = []
                for i in range(n):
                    if(color[i] == now_color):
                        color_to_site[vertex_color_num] = i
                        site_to_color[i] = vertex_color_num
                        vertex_color_num += 1
                        if(value_type[i] == 'B'):
                            now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY)
                        elif(value_type[i] == 'I'):
                            now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER)
                        else:
                            now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS)
                        x.append(now_val)
                    
                #Set up the objective function and optimization objective (maximization/minimization), only optimizing the variables in this block
                objsum = 0
                objtemp = 0
                for i in range(n):
                    if(color[i] == now_color):
                        objsum += x[site_to_color[i]] * coefficient[i]
                    else:
                        objtemp += ansx[i] * coefficient[i]
                if(obj_type == 'maximize'):
                    model.setObjective(objsum, GRB.MAXIMIZE)
                else:
                    model.setObjective(objsum, GRB.MINIMIZE)
                #Add m constraints, only adding those constraints that involve variables in this block
                for i in range(m): 
                    flag = 0
                    constr = 0
                    for j in range(k[i]):
                        if(color[site[i][j]] == now_color):
                            flag = 1
                            constr += x[site_to_color[site[i][j]]] * value[i][j]
                        else:
                            constr += ansx[site[i][j]] * value[i][j]
                    if(flag):
                        if(constraint_type[i] == 1):
                            model.addConstr(constr <= constraint[i])
                        elif(constraint_type[i] == 2):
                            model.addConstr(constr >= constraint[i])
                        else:
                            model.addConstr(constr == constraint[i])
                
                #Set the maximum solving time
                model.setParam('TimeLimit', min(max(time_limit - (time.time() - begin_time), 0), max_turn_time))
                #Optimize
                model.optimize()
                
                try:
                    #Calculate the current objective value
                    temp = model.ObjVal + objtemp
                    print(f"The current objective value is: {temp}")
                    bestX = []
                    for i in range(vertex_color_num):
                        bestX.append(x[i].X)
                    #print(bestX)

                    #Update the current best solution and best ans
                    if(obj_type == 'maximize'):
                        if(temp > ans):
                            GAP = model.MIPGap
                            for i in range(vertex_color_num):
                                ansx[color_to_site[i]] = bestX[i]
                            ans = temp
                    else:
                        if(temp < ans):
                            GAP = model.MIPGap
                            for i in range(vertex_color_num):
                                ansx[color_to_site[i]] = bestX[i]
                            ans = temp
                            
                    now_sol.append(ans)
                    now_time.append(time.time() - begin_time)
                    
                except:
                    print("Cant't optimize more~~")
                    # new_ansx = {}
                    # for i in range(len(ansx)):
                    #     new_ansx[num_to_value[i]] = ansx[i]
                    # with open(pickle_path + '/' + (os.path.split(lp_file)[1])[:-3] + '.pickle', 'wb') as f:
                    #     pickle.dump([new_ansx, ans], f)
                    # return ans, time.time()-begin_time
            
        new_ansx = {}
        for i in range(len(ansx)):
            new_ansx[num_to_value[i]] = ansx[i]

        for _ in range(len(now_sol)):
            result_list.append((result_list[0][0] + now_time[_], now_sol[_]))

        self.end()
        return 0, 0, 0 


class NALNS(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        self.choose = kwargs.get("choose") or 0.5
        
        ... # tackle parameters

    def work(self, input: Cansol2S, result_list: list):
        
        self.begin()
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type = split_problem_NALNS(self.instance)

        new_sol = []
        tmp = gp.read(self.instance)
        for var in tmp.getVars():
            new_sol.append(input.cansol[var.VarName])
        
        
        time_limit = self.time_limit

        new_site = []
        new_value = []
        new_constraint = np.zeros(m)
        new_constraint_type = np.zeros(m, int)
        for i in range(m):
            new_site.append(np.zeros(k[i], int))
            new_value.append(np.zeros(k[i]))
            for j in range(k[i]):
                new_site[i][j] = site[i][j]
                new_value[i][j] = value[i][j]
            new_constraint[i] = constraint[i]
            new_constraint_type[i] = constraint_type[i]
        
        new_coefficient = np.zeros(n)
        new_lower_bound = np.zeros(n)
        new_upper_bound = np.zeros(n)
        new_value_type = np.zeros(n, int)
        new_new_sol = np.zeros(n)
        for i in range(n):
            new_coefficient[i] = coefficient[i]
            new_lower_bound[i] = lower_bound[i]
            new_upper_bound[i] = upper_bound[i]
            if(value_type[i] == 'B'):
                new_value_type[i] = 0
            elif(value_type[i] == 'C'):
                new_value_type[i] = 1
            else:
                new_value_type[i] = 2
            new_new_sol[i] = new_sol[i]

        now_instance = (n, m, k, new_site, new_value, new_constraint, new_constraint_type, new_coefficient, obj_type, new_lower_bound, new_upper_bound, new_value_type, new_new_sol)
        now_sol, now_time, now_gap = greedy_one_NALNS(now_instance, time_limit, self.choose)

        for _ in range(len(now_sol)):
            result_list.append((result_list[0][0] + now_time[_], now_sol[_]))

        self.end()
        
        return 0, 0, 0 

class Gurobi(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get('time_limit') or 10
        self.log = []
        self.benchmark_path = kwargs.get('benchmark_path') or 0
        ... # tackle parameters

    def work(self, input: Cansol2S, result_list: list):
        self.begin()
        
        model = gp.read(self.instance)
        model.setParam('TimeLimit', self.time_limit)

        for var in model.getVars():
            var.Start = input.cansol[var.VarName]

        log = []
        def my_callback(model, where):
            if where == GRB.Callback.MIP:
                runtime = model.cbGet(GRB.Callback.RUNTIME)
                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                log.append((runtime, obj_best))


        model.optimize(my_callback)

        for _ in range(len(log)):
            result_list.append((result_list[0][0] + log[_][0], log[_][1]))

        if self.benchmark_path != 0:
            
            instance_name = os.path.basename(self.instance)
            tmp = re.match(r"(.*)\.lp", instance_name)
            tmp = tmp.group(1)
            pickle_path = os.path.join(self.benchmark_path, tmp + '.pickle')
            solution = {}
            for var in model.getVars():
                solution[var.VarName] = var.X
            with open(pickle_path, 'wb') as f:
                pickle.dump([solution, model.MIPGap], f)
        
        self.end()
        
        return model.MIPGap, model.ObjVal, model.ModelSense # must be it!
        

class ACP(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get('time_limit') or 10
        self.block = kwargs.get('block') or 2
        self.max_turn_ratio = kwargs.get('max_turn_ratio') or 0.1
        ... # tackle parameters

    def work(self, input: Cansol2S, result_list: list):
        self.begin()

        '''
        Run LNS (Large Neighborhood Search), passing in the lp file. 'block' is the number of blocks (default 2), 'time_limit' is the total running time limit (default 4000), and 'max_turn_ratio' is the maximum running time ratio for each turn (default 0.1).
        '''
        #Set KK as the initial number of blocks and PP as the selected number of blocks to optimize after dividing the constraints into KK blocks
        KK = self.block
        PP = 1
        max_turn = 5
        epsilon = 0.01
        #Retrieve the problem model after splitting and create a new folder named "ACP_Pickle"
        max_turn_ratio = self.max_turn_ratio
        time_limit = self.time_limit
        max_turn_time = max_turn_ratio * time_limit
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num = split_problem_ACP(self.instance)    
        # Build a variable dictionar
        num_to_value = {value : key for key, value in value_to_num.items()}

        #Get the start time
        begin_time = time.time()
        
        #Initialize the initial solution and initial answer, where the initial solution is the worst initial solution
        
        ans, ansx = input.objval, []
        
        tmp = gp.read(self.instance)
        for var in tmp.getVars():
            ansx.append(input.cansol[var.VarName])
            
        print(f"初始解目标值为：{ans}")
        
        #Constraint block labels, where cons_color[i] represents which block the i-th constraint belongs to
        cons_color = np.zeros(m, int)
        
        #Initialize the initial solution and initial answer, where the initial solution is the worst initial solution
        last = ans
        
        #Initialize the number of rounds below the threshold, which can be either 0 or 1 with little difference.
        turn = 1
        #Continue to the next iteration as long as it hasn't reached the maximum running time

        now_sol, now_time = [], []
 
        while(time.time() - begin_time <= time_limit):
            print("KK = ", KK)
            print("PP = ", PP)
            #Randomly divide the decision variables into KK blocks
            for i in range(m):
                cons_color[i] = random.randint(1, KK)
            now_cons_color = 1
            #Set all decision variables involved in a randomly selected constraint block as variables to be optimized
            #color[i] = 1 indicates that the i-th decision variable is selected as a variable to be optimized
            color = np.zeros(n, int)
            now_color = 1
            color_num = 0
            for i in range(m):
                if(PP == 1 and cons_color[i] == now_cons_color):
                    for j in range(k[i]):
                        color[site[i][j]] = 1
                        color_num += 1
                if(PP > 1 and cons_color[i] != now_cons_color):
                    for j in range(k[i]):
                        color[site[i][j]] = 1
                        color_num += 1
            #site_to_color[i]represents which decision variable is the i-th decision variable in this block
            #color_to_site[i]represents which decision variable is mapped to the i-th decision variable in this block
            #vertex_color_num represents the number of decision variables in this block currently
            site_to_color = np.zeros(n, int)
            color_to_site = np.zeros(n, int)
            vertex_color_num = 0
            #Define the model to solve
            model = gp.Model("ACP")
            x = []
            for i in range(n):
                if(color[i] == now_color):
                    color_to_site[vertex_color_num] = i
                    site_to_color[i] = vertex_color_num
                    vertex_color_num += 1
                    if(value_type[i] == 'B'):
                        now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY)
                    elif(value_type[i] == 'I'):
                        now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER)
                    else:
                        now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS)
                    x.append(now_val)
            #Define decision variables x[]
            #x = model.addMVar((vertex_color_num), lb = 0, ub = 1, vtype = GRB.BINARY)  #lb is the lower bound for the variable， ub is the upper bound for the variables
            #Set up the objective function and optimization objective (maximization/minimization), only optimizing the variables selected for optimization
            objsum = 0
            objtemp = 0
            for i in range(n):
                if(color[i] == now_color):
                    objsum += x[site_to_color[i]] * coefficient[i]
                else:
                    objtemp += ansx[i] * coefficient[i]
            if(obj_type == 'maximize'):
                model.setObjective(objsum, GRB.MAXIMIZE)
            else:
                model.setObjective(objsum, GRB.MINIMIZE)
            #Add m constraints, only adding those constraints that involve variables in this block
            for i in range(m): 
                flag = 0
                constr = 0
                for j in range(k[i]):
                    if(color[site[i][j]] == now_color):
                        flag = 1
                        constr += x[site_to_color[site[i][j]]] * value[i][j]
                    else:
                        constr += ansx[site[i][j]] * value[i][j]
                if(flag):
                    if(constraint_type[i] == 1):
                        model.addConstr(constr <= constraint[i])
                    elif(constraint_type[i] == 2):
                        model.addConstr(constr >= constraint[i])
                    else:
                        model.addConstr(constr == constraint[i])
            
            #Set the maximum solving time
            model.setParam('TimeLimit', min(max(time_limit - (time.time() - begin_time), 0), max_turn_time))
            #Optimize
            model.optimize()
            
            try:
                #Calculate the current objective value
                temp = model.ObjVal + objtemp
                print(f"当前目标值为：{temp}")
                bestX = []
                for i in range(vertex_color_num):
                    bestX.append(x[i].X)
                #print(bestX)

                if(obj_type == 'maximize'):
                    #Update the current best solution and best ans
                    if(temp > ans):
                        for i in range(vertex_color_num):
                            ansx[color_to_site[i]] = bestX[i]
                        ans = temp
                    #Adaptive block number change
                    if((ans - last <= epsilon * ans)):
                        turn += 1
                        if(turn == max_turn):
                            if(KK > 2 and PP == 1):
                                KK -= 1
                            else:
                                KK += 1
                                PP += 1
                            turn = 1
                    else:
                        turn = 0
                else:
                    if(temp < ans):
                        for i in range(vertex_color_num):
                            ansx[color_to_site[i]] = bestX[i]
                        ans = temp
                    #Adaptive block number change
                    if((last - ans <= epsilon * ans)):
                        turn += 1
                        if(turn == max_turn):
                            if(KK > 2 and PP == 1):
                                KK -= 1
                            else:
                                KK += 1
                                PP += 1
                            turn = 1
                    else:
                        turn = 0
                    
                now_sol.append(ans)
                now_time.append(time.time() - begin_time)
                                   
                if(model.MIPGap != 0):
                    if(KK == 2 and PP > 1):
                        KK -= 1
                        PP -= 1
                    else:
                        KK += 1
                    turn = 0
                last = ans
            except:
                try:
                    model.computeIIS()
                    if(KK > 2 and PP == 1):
                        KK -= 1
                    else:
                        KK += 1
                        PP += 1
                    turn = 1
                except:
                    if(KK == 2 and PP > 1):
                        KK -= 1
                        PP -= 1
                    else:
                        KK += 1
                    turn = 0                    
                print("This turn can't improve more")
        new_ansx = {}
        for i in range(len(ansx)):
            new_ansx[num_to_value[i]] = ansx[i]

        for _ in range(len(now_sol)):
            result_list.append((result_list[0][0] + now_time[_], now_sol[_]))

        self.end()
        return 0, 0, 0  
        
class SCIP(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get('time_limit') or 10
        
        ... # tackle parameters

    def work(self, input: Cansol2S, result_list: list):
        self.begin()

        model = scp.Model()
        model.setParam('limits/time', self.time_limit)
        #m1.hideOutput(True)
        model.setParam('randomization/randomseedshift', 0)
        model.setParam('randomization/lpseed', 0)
        model.setParam('randomization/permutationseed', 0)
        model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)#MIP focus
        model.readProblem(self.instance)
        new_sol = model.createSol()
        for var in model.getVars():
            model.setSolVal(new_sol, var, input.cansol[var.name])
        model.addSol(new_sol)

        log_time = []
        log_val = []
        def get_val_time(model, event):
            log_time.append(model.getSolvingTime())
            log_val.append(model.getPrimalbound())
            
        model.attachEventHandlerCallback(get_val_time, [SCIP_EVENTTYPE.BESTSOLFOUND])
        model.optimize()
        log_val.append(model.getPrimalbound())
        log = []
        for i in range(len(log_time)):
            log.append((log_time[i], log_val[i + 1]))
        
        for _ in range(len(log)):
            result_list.append((result_list[0][0] + log[_][0], log[_][1]))
            
        self.end()

#        return model.getGap(), model.getObjVal()
        return 0, 0, 0
