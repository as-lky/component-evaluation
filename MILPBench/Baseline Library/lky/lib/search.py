import os
import numpy as np
import pyscipopt as scp

from help.LIH.help import greedy_one as greedy_one_LIH, split_problem as split_problem_LIH
from help.MIH.help import greedy_one as greedy_one_MIH, split_problem as split_problem_MIH

from pyscipopt import SCIP_PARAMSETTING
from typing import Type
from mod import Component, Modify2Search, Cansol, Scores





class Search(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "scip":
            return super().__new__(SCIP, device, taskname, instance, sequence_name, *args, **kwargs)
        elif component == "gurobi":
            return super().__new__(Gurobi, device, taskname, instance, sequence_name, *args, **kwargs)
        elif component == "LIH":
            return super().__new__(LIH, device, taskname, instance, sequence_name, *args, **kwargs)
        else:
            raise ValueError("Search component type is not defined")
        
    def __init__(self, device, taskname, instance, sequence_name):
        super.__init__(device, taskname, instance, sequence_name)
        
        
class LIH(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super.__init__(device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 3600)
        ... # tackle parameters

    def work(self, input: Type[Cansol]):
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
        super.__init__(device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 3600)
        ... # tackle parameters

    def work(self, input: Type[Cansol]):
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
        
        
        
class Gurobi(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super.__init__(device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self, input: Type[Modify2Search]):
        ...
        
class SCIP(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super.__init__(device, taskname, instance, sequence_name)
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
        
        ... # tackle parameters

    def work(self, input: Type[Scores]):

        self.begin()
        m1 = scp.Model()
        m1.setParam('limits/time', 1000)
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
