import os
import numpy as np
import pyscipopt as scp
import time
import random
import gurobipy as gp

from .help.LIH.help import greedy_one as greedy_one_LIH, split_problem as split_problem_LIH
from .help.MIH.help import greedy_one as greedy_one_MIH, split_problem as split_problem_MIH
from .help.NALNS.help import greedy_one as greedy_one_NALNS, split_problem as split_problem_NALNS
from .help.LNS.help import split_problem as split_problem_LNS
from .mod import Component, Modify2Search, Cansol, MScores

from pyscipopt import SCIP_PARAMSETTING
from typing import Self, Type, cast
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
        else:
            raise ValueError("Search component type is not defined")

        return super().__new__( cast(type[Self], cls) )
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)

    def work(self, input: Modify2Search):...
        
class LIH(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 3600)
        ... # tackle parameters

    def work(self, input: Cansol):
        self.begin()
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type = split_problem_LIH(self.instance)
        new_sol = input.cansol
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
        
        now_sol, now_time = greedy_one_LIH(now_instance, time_limit)
        self.end()
        return now_sol, now_time # TODO: modify return
 
class MIH(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 3600)
        ... # tackle parameters

    def work(self, input: Cansol):
        self.begin()
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
            new_new_sol[i] = input.cansol[i]

        now_instance = (n, m, k, new_site, new_value, new_constraint, new_constraint_type, new_coefficient, obj_type, new_lower_bound, new_upper_bound, new_value_type, new_new_sol)
        now_sol, now_time = greedy_one_MIH(now_instance, time_limit)
        # print(now_sol)
        # print(now_time)
        self.end()
        return now_sol, now_time # TODO: modify return
        
class LNS(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 3600)
        self.block = kwargs.get("block", 4)
        self.max_turn_ratio = kwargs.get("max_turn_ratio", 0.01)
        
        ... # tackle parameters

    def work(self, input: Cansol):
        self.begin()
        
        time_limit = self.time_limit
        max_turn_time = self.max_turn_ratio * self.time_limit
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num = split_problem_LNS(self.instance)
        # Build a variable dictionar
        num_to_value = {value : key for key, value in value_to_num.items()}
        #Set KK as the initial number of blocks, and PP as the selected number of blocks to optimize after dividing the constraints into KK blocks
        KK = self.block
        
        ans, ansx = input.objval, input.cansol
        # Here the ans is the objective value, ansx is the sol of index(like ans in LIH MIH)
        print(f"Initial objective: {ans}")
                
        begin_time = time.time()
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
                            for i in range(vertex_color_num):
                                ansx[color_to_site[i]] = bestX[i]
                            ans = temp
                    else:
                        if(temp < ans):
                            for i in range(vertex_color_num):
                                ansx[color_to_site[i]] = bestX[i]
                            ans = temp
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

        self.end()
        return ans, time.time() - begin_time # TODO: modify return

class NALNS(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 3600)
        
        ... # tackle parameters

    def work(self, input: Cansol):
        
        self.begin()
        n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type = split_problem_NALNS(self.instance)
        new_sol = input.cansol
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
        now_sol, now_time = greedy_one_NALNS(now_instance, time_limit)
        # print(now_sol)
        # print(now_time)
        self.end()
        
        return now_sol, now_time # TODO: modify return
         

class Gurobi(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self, input: Type[Modify2Search]):
        ...
        
class SCIP(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        if taskname == "IP": 
            dhp = 1
        elif taskname == "IS":
            dhp = 15 
        elif taskname == "WA":
            dhp = 5 
        elif taskname == "CA":
            dhp = 10
        else :
            dhp = 0 # default hyperparameter
                
        self.delta = kwargs.get('delta', dhp)
        self.time_limit = kwargs.get('time_limit', 10)
        
        ... # tackle parameters

    def work(self, input: MScores):

        self.begin()
        m1 = scp.Model()
        m1.setParam('limits/time', self.time_limit)
        #m1.hideOutput(True)
        m1.setParam('randomization/randomseedshift', 0)
        m1.setParam('randomization/lpseed', 0)
        m1.setParam('randomization/permutationseed', 0)
        m1.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)#MIP focus

        instance_name = os.path.basename(self.instance)
        log_path = f'./logs/{self.taskname}/{self.sequence_name}/work/{instance_name}.log'
        m1.setLogfile(log_path)
        m1.readProblem(self.instance)

        #trust region method implemented by adding constraints
        m1_vars = m1.getVars()
        var_map1 = {}
        for v in m1_vars:  # get a dict (variable map), varname:var clasee
            var_map1[v.name] = v
        alphas = []
        for i in range(len(input.scores)):
            tar_var = var_map1[input.scores[i][1]]  # target variable <-- variable map
            x_star = input.scores[i][3]  # 1,0,-1, decide whether to fix
            if x_star < 0: # -1 no need to fix
                continue
            tmp_var = m1.addVar(f'alp_{tar_var}_{i}', 'C')
            alphas.append(tmp_var)
            m1.addCons(tmp_var >= tar_var - x_star, f'alpha_up_{i}')
            m1.addCons(tmp_var >= x_star - tar_var, f'alpha_down_{i}')
        m1.addCons(scp.quicksum(ap for ap in alphas) <= self.delta, 'sum_alpha')
        m1.optimize()
        self.end() # TODO: add return
        
        return m1.getGap()