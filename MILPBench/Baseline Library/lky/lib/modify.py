import torch
from typing import Type, cast, Self
from .mod import Component, Predict2Modify, Modify2Search, Cantsol, Cansol2M, Cansol2S
from .help.GCN.helper import get_a_new2
from .help.NEURALDIVING.test import Gurobi_solver 
from pyscipopt import SCIP_PARAMSETTING
import gurobipy as gp
import pyscipopt as scp        
import os


class Modify(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "np":
            cls = Np
        elif component == 'rs':
            cls = Rs
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
        
        

class Rs(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        
        # tackle parameters    

    def work(self, input: Cantsol) -> Cansol2S:
        self.begin()
        # scores a list of (index, VariableName, Prob)
        
        
        model = gp.read(self.instance)
        value_to_num = {}
        num_to_value = {}
        value_to_type = {}
        value_num = 0
        #N represents the number of decision variables
        #M represents the number of constraints
        #K [i] represents the number of decision variables in the i-th constraint
        #Site [i] [j] represents which decision variable is the jth decision variable of the i-th constraint
        #Value [i] [j] represents the coefficient of the jth decision variable of the i-th constraint
        #Constraint [i] represents the number to the right of the i-th constraint
        #Constrict_type [i] represents the type of the i-th constraint, 1 represents<, 2 represents>, and 3 represents=
        #Coefficient [i] represents the coefficient of the i-th decision variable in the objective function
        n = model.NumVars
        m = model.NumConstrs
        k = []
        site = []
        value = []
        constraint = []
        constraint_type = []
        for cnstr in model.getConstrs():
            if(cnstr.Sense == '<'):
                constraint_type.append(1)
            elif(cnstr.Sense == '>'):
                constraint_type.append(2) 
            else:
                constraint_type.append(3) 
            
            constraint.append(cnstr.RHS)


            now_site = []
            now_value = []
            row = model.getRow(cnstr)
            k.append(row.size())
            for i in range(row.size()):
                if(row.getVar(i).VarName not in value_to_num.keys()):
                    value_to_num[row.getVar(i).VarName] = value_num
                    num_to_value[value_num] = row.getVar(i).VarName
                    value_num += 1
                now_site.append(value_to_num[row.getVar(i).VarName])
                now_value.append(row.getCoeff(i))
            site.append(now_site)
            value.append(now_value)

        coefficient = {}
        lower_bound = {}
        upper_bound = {}
        value_type = {}
        for val in model.getVars():
            if(val.VarName not in value_to_num.keys()):
                value_to_num[val.VarName] = value_num
                num_to_value[value_num] = val.VarName
                value_num += 1
            coefficient[value_to_num[val.VarName]] = val.Obj
            lower_bound[value_to_num[val.VarName]] = val.LB
            upper_bound[value_to_num[val.VarName]] = val.UB
            value_type[value_to_num[val.VarName]] = val.Vtype

        #1 minimize, -1 maximize
        obj_type = model.ModelSense
        time_limit = self.time_limit

        now_sol = input.scores 
        for i in range(n):
            if(value_type[i] != 'C'):
                now_sol[i] = int(now_sol[i] + 0.5)
            now_sol[i] = min(now_sol[i], upper_bound[i])
            now_sol[i] = max(now_sol[i], lower_bound[i])
       
        
        for turn in range(11):
            choose = []
            rate = (int)(0.1 * turn * n)
            for i in range(n):
                if(select[i] >= new_select[rate]):
                    choose.append(1)
                else:
                    choose.append(0)
            #print(0.1 * turn, sum(choose) / n)
            flag, sol, obj = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, choose, lower_bound, upper_bound, value_type)
            if(flag == 1):
                add_flag = 1
                result.append(obj)
                break

        
        
        
        
        self.end()
        return 
        

    
        
        
class Default(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        # tackle parameters    

    def work(self, input: Cansol2M) -> Cansol2S:
        self.begin()
        self.end()
        return Cansol2S(input.objval, input.cansol, input.gap)
        