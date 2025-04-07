import torch
from typing import Type, cast, Self
from .mod import Component, Predict2Modify, Modify2Search, Cantsol, Cansol2M, Cansol2S, INFEASIBLEERROR
from .help.NEURALDIVING.read_lp import get_a_new2
from .help.NEURALDIVING.test import Gurobi_solver 
from pyscipopt import SCIP_PARAMSETTING
import numpy as np
import gurobipy as gp
import pyscipopt as scp        
import os


def log(any, txt):
    with open(txt, 'a')as f:
        f.writelines(str(any))
        f.writelines('\n')

class Modify(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "np":
            cls = Np
        elif component == 'sr':
            cls = Sr
        elif component == "nr":
            cls = Nr
        elif component == "default":
            cls = Default
        else:
            raise ValueError("Modify component type is not defined")
        return super().__new__( cast(type[Self], cls) )
        
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)
            
    def work(self, input: Predict2Modify) -> Modify2Search:...
        
class Np(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        
        if taskname == "IP": 
            dhp = (400, 5, 1)
        elif taskname == "IS":
            dhp = (300, 300, 15) 
        elif taskname == "WA":
            dhp = (0, 600, 5) 
        elif taskname == "CA":
            dhp = (400, 0, 10)
        else :
            dhp = (0, 0, 0)
        
        self.k0 = kwargs.get("k0", dhp[0])
        self.k1 = kwargs.get("k1", dhp[1])
        self.delta = kwargs.get("delta", dhp[2])
        # tackle parameters    

    def work(self, input: Cantsol) -> Cansol2S:
        self.begin()
        
        constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value=get_a_new2(self.instance)
        select = input.select.to('cpu').detach().numpy() 

        scores = []
        for i in range(n):
            if value_type[i] == "BINARY":
                scores.append([i, num_to_value[i], select[i], -1])
        # get a list of (index, name, Prob, -1)

        scores.sort(key=lambda x:x[2], reverse=True)

        # fixing variable picked by confidence scores
        count1 = 0
        for i in range(len(scores)):
            if count1 < self.k1:
                scores[i][3] = 1
                count1 += 1

        scores.sort(key=lambda x: x[2], reverse=False)
        count0 = 0
        for i in range(len(scores)):
            if count0 < self.k0:
                scores[i][3] = 0
                count0 += 1
        
        m1 = scp.Model()
        m1.setParam('limits/time', self.time_limit)
        #m1.hideOutput(True)
        m1.setParam('randomization/randomseedshift', 0)
        m1.setParam('randomization/lpseed', 0)
        m1.setParam('randomization/permutationseed', 0)
        m1.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)#MIP focus

        instance_name = os.path.basename(self.instance)
        
        sn = ""
        for _ in self.sequence_name:
            sn += _ + "_"

        
        log_path = f'./logs/work/{self.taskname}/{sn}/{instance_name}.log'
        m1.setLogfile(log_path)
        m1.readProblem(self.instance)

        #trust region method implemented by adding constraints
        m1_vars = m1.getVars()
        var_map1 = {}
        for v in m1_vars:  # get a dict (variable map), varname:var clasee
            var_map1[v.name] = v
        alphas = []
        for i in range(len(scores)):
            tar_var = var_map1[scores[i][1]]  # target variable <-- variable map
            x_star = scores[i][3]  # 1,0,-1, decide whether to fix
            if x_star < 0: # -1 no need to fix
                continue
            tmp_var = m1.addVar(f'alp_{tar_var}_{i}', 'C')
            alphas.append(tmp_var)
            m1.addCons(tmp_var >= tar_var - x_star, f'alpha_up_{i}')
            m1.addCons(tmp_var >= x_star - tar_var, f'alpha_down_{i}')
        m1.addCons(scp.quicksum(ap for ap in alphas) <= self.delta, 'sum_alpha')
        m1.optimize()
        if m1.getNSols() == 0:
            sn = ""
            for _ in self.sequence_name:
                sn += _ + "_"
            des = f'./logs/work/{self.taskname}/{sn}/result.txt'
            log("ERROR", des)
            log("MODIFY INFEASIBLE", des)
            raise INFEASIBLEERROR("Modify infeasible")
        
        cansol = {}
        for var in m1.getVars():
            cansol[var.name] = m1.getVal(var)
        
        self.end()
        return Cansol2S(m1.getGap(), cansol, m1.getObjVal())
        
        

class Sr(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        
        # tackle parameters    

    def work(self, input: Cantsol) -> Cansol2S:
        self.begin()
        model = gp.read(self.instance)
        constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value=get_a_new2(self.instance)

        time_limit = self.time_limit

        new_select = input.select.to('cpu').detach().numpy() 
        new_select.sort()

        now_sol = input.logits.to('cpu').detach().numpy() 
         
        for i in range(n):
            if(value_type[i] != 'C'):
                now_sol[i] = int(now_sol[i] + 0.5)
            now_sol[i] = min(now_sol[i], upper_bound[i])
            now_sol[i] = max(now_sol[i], lower_bound[i])
       
        result_pair = (0, 0, 0)
        add_flag = 0
        for turn in range(11):
            choose = []
            rate = (int)(0.1 * turn * n)
            for i in range(n):
                if(input.select[i] >= new_select[rate]):
                    choose.append(0)
                else:
                    choose.append(1)
            #print(0.1 * turn, sum(choose) / n)
            flag, sol, obj, gap = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, choose, lower_bound, upper_bound, value_type)
            if(flag == 1):
                add_flag = 1
                result_pair = (sol, obj, gap)
                break

        if add_flag == 0:
            sn = ""
            for _ in self.sequence_name:
                sn += _ + "_"
            des = f'./logs/work/{self.taskname}/{sn}/result.txt'
            log("ERROR", des)
            log("MODIFY INFEASIBLE", des)
            raise INFEASIBLEERROR("Modify infeasible")
        
        cansol = {}
        for i in range(n):
            cansol[num_to_value[i]] = result_pair[0][i]
        
        self.end()
        return Cansol2S(result_pair[1], cansol, result_pair[2])
        
        
class Nr(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        
        # tackle parameters    

    def work(self, input: Cantsol) -> Cansol2S:
        self.begin()
        constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value=get_a_new2(self.instance)

        color = np.zeros(n) # 1: discard
        
        now_sol = input.logits.to('cpu').detach().numpy() 

        for i in range(n):
            if(value_type[i] != 'C'):
                now_sol[i] = int(now_sol[i] + 0.5)
            now_sol[i] = min(now_sol[i], upper_bound[i])
            now_sol[i] = max(now_sol[i], lower_bound[i])
 
 
        F = 0
        result_pair = (0, 0, 0)
        for LL in range(3): # 3 times repair. If can't then fail
            discard = []
            for j in range(m):
                constr = 0
                flag = 0
                for l in range(k[j]):
                    if color[site[j][l]] == 1:
                        flag = 1
                        break
                    else:
                       constr += now_sol[site[j][l]] * value[j][l]
                if flag == 1:
                    for l in range(k[j]):
                        if color[site[j][l]] == 0:
                            discard.append(site[j][l])
                    continue
                if(constraint_type[j] == 1):
                    if(constr > constraint[j]):
                        for l in range(k[j]):
                            if color[site[j][l]] == 0:
                                discard.append(site[j][l])
                else:
                    if(constr < constraint[j]):
                        for l in range(k[j]):
                            if color[site[j][l]] == 0:
                                discard.append(site[j][l])
            for i in discard:
                color[i] = 1
                
            flag, sol, obj, gap = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, self.time_limit, obj_type, now_sol, color, lower_bound, upper_bound, value_type)

            if flag == 1:
                result_pair = (sol, obj, gap)
                F = 1
                break
        
        if F == 0:
            sn = ""
            for _ in self.sequence_name:
                sn += _ + "_"
            des = f'./logs/work/{self.taskname}/{sn}/result.txt'
            log("ERROR", des)
            log("MODIFY INFEASIBLE", des)
            raise INFEASIBLEERROR("Modify infeasible")
        
        cansol = {}
        for i in range(n):
            cansol[num_to_value[i]] = result_pair[0][i]

        self.end()
        return Cansol2S(result_pair[1], cansol, result_pair[2])
        
        
        
class Default(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        # tackle parameters    

    def work(self, input: Cansol2M) -> Cansol2S:
        self.begin()
        self.end()
        return Cansol2S(input.objval, input.cansol, input.gap)
        