import torch
from typing import Type, cast, Self
from .mod import Component, Predict2Modify, Modify2Search, Cantsol, Cansol2M, Cansol2S
from .help.GCN.helper import get_a_new2
from pyscipopt import SCIP_PARAMSETTING
import pyscipopt as scp        
import os


class Modify(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "np":
            cls = Np
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
        
        A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(self.instance)
        scores = input.scores

        binary_name = [scores[i][1] for i in b_vars]
        # get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(scores)):
            type = "C"
            if scores[i][1] in binary_name:
                type = 'BINARY'
            scores[i] += [-1, type]

        scores=[x for x in scores if x[4]=='BINARY'] # get binary

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
        log_path = f'./logs/work/{self.taskname}/{self.sequence_name}/{instance_name}.log'
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
        
        self.end()
        cansol = {}
        for var in m1.getVars():
            cansol[var.name] = m1.getVal(var)
        
        return Cansol2S(m1.getGap(), cansol, m1.getObjVal())
        
class Default(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        # tackle parameters    

    def work(self, input: Cansol2M) -> Cansol2S:
        self.begin()
        self.end()
        return Cansol2S(input.objval, input.cansol, input.gap)
        