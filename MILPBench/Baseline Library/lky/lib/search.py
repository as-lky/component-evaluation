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
from .mod import Component, Modify2Search, Cansol2S

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
        self.time_limit = kwargs.get("time_limit") or 10
        ... # tackle parameters

    def work(self, input: Cansol2S):
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
        
        now_sol, now_time, now_gap = greedy_one_LIH(now_instance, time_limit)
        self.end()
        return now_gap # TODO: modify return
 
class MIH(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        ... # tackle parameters

    def work(self, input: Cansol2S):
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
        now_sol, now_time, now_gap = greedy_one_MIH(now_instance, time_limit)
        # print(now_sol)
        # print(now_time)
        self.end()
        return now_gap, now_sol # TODO: modify return
        
class LNS(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        self.block = kwargs.get("block") or 4
        self.max_turn_ratio = kwargs.get("max_turn_ratio") or 0.01
        
        ... # tackle parameters

    def work(self, input: Cansol2S):
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
        GAP = input.gap  # TODO : check LNS gap
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
        return GAP, ans # TODO: modify return

class NALNS(Search):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        
        ... # tackle parameters

    def work(self, input: Cansol2S):
        
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
        now_sol, now_time, now_gap = greedy_one_NALNS(now_instance, time_limit)
        # print(now_sol)
        # print(now_time)
        self.end()
        
        return now_gap, now_sol # TODO: modify return
         

class Gurobi(Search): # solver
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get('time_limit') or 10
        ... # tackle parameters

    def work(self, input: Cansol2S):
        self.begin()
        
        model = gp.read(self.instance)
        model.setParam('TimeLimit', self.time_limit)

        for var in model.getVars():
            var.Start = input.cansol[var.VarName]
        model.optimize()
    
        self.end()
        
        return model.MIPGap, model.ObjVal
        
        
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
                
        self.delta = kwargs.get('delta') or dhp
        self.time_limit = kwargs.get('time_limit') or 10
        
        ... # tackle parameters

    def work(self, input: Cansol2S):
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
        model.optimize()
        
        self.end()

        return model.getGap(), model.getObjVal()