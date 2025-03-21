import torch
import os
import pyscipopt as scp

from pyscipopt import SCIP_PARAMSETTING
from typing import Type
from mod import Component, Modify2Search


class Search(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "scip":
            return super().__new__(SCIP, device, taskname, instance, sequence_name, *args, **kwargs)
        elif component == "gurobi":
            return super().__new__(Gurobi, device, taskname, instance, sequence_name, *args, **kwargs)
        else :
            raise ValueError("Search component type is not defined")
        
    def __init__(self, device, taskname, instance, sequence_name):
        super.__init__(device, taskname, instance, sequence_name)
        
class Gurobi(Search):
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super.__init__(device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self, input: Type[Modify2Search]):
        ...
        
class SCIP(Search):
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super.__init__(device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self, input: Type[Modify2Search]):

        m1 = scp.Model()
        m1.setParam('limits/time', 1000)
        #m1.hideOutput(True)
        m1.setParam('randomization/randomseedshift', 0)
        m1.setParam('randomization/lpseed', 0)
        m1.setParam('randomization/permutationseed', 0)
        m1.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)#MIP focus
        instance_name = os.path.basename(self.instance)
        log_path = f'./logs/{self.taskname}/{self.sequence_name}/{instance_name}.log'
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
            x_star = scores[i][3]  # 1,0,-1, decide whether to fix
            if x_star < 0:
                continue
            tmp_var = m1.addVar(f'alp_{tar_var}_{i}', 'C')
            alphas.append(tmp_var)
            m1.addCons(tmp_var >= tar_var - x_star, f'alpha_up_{i}')
            m1.addCons(tmp_var >= x_star - tar_var, f'alpha_down_{i}')
        m1.addCons(scp.quicksum(ap for ap in alphas) <= delta, 'sum_alpha')
        m1.optimize()
