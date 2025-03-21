import os
import copy
import time
import gurobipy as gp
import numpy as np
from gurobipy import GRB, read, Model
import pandas as pd

def initial_solution(n, m, k, site, value, constraint, constraint_type, lower_bound, upper_bound, value_type):
    model = Model("Gurobi")
    x = []
    for i in range(n):
        if(value_type[i] == 'B'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
        elif(value_type[i] == 'C'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
        else:
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
    model.setObjective(0, GRB.MAXIMIZE)
    for i in range(m):
        constr = 0
        for j in range(k[i]):
            constr += x[site[i][j]] * value[i][j]

        if(constraint_type[i] == 1):
            model.addConstr(constr <= constraint[i])
        elif(constraint_type[i] == 2):
            model.addConstr(constr >= constraint[i])
        else:
            model.addConstr(constr == constraint[i])
    model.optimize()
    new_sol = np.zeros(n)
    for i in range(n):
        if(value_type[i] == 'C'):
            new_sol[i] = x[i].X
        else:
            new_sol[i] = (int)(x[i].X)
        
    return new_sol

def initial_LP_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type):
    begin_time = time.time()
    model = Model("Gurobi")
    x = []
    for i in range(n):
        x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
    coeff = 0
    for i in range(n):
        coeff += x[i] * coefficient[i]
    if(obj_type == -1):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    for i in range(m):
        constr = 0
        for j in range(k[i]):
            constr += x[site[i][j]] * value[i][j]

        if(constraint_type[i] == 1):
            model.addConstr(constr <= constraint[i])
        elif(constraint_type[i] == 2):
            model.addConstr(constr >= constraint[i])
        else:
            model.addConstr(constr == constraint[i])
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    model.optimize()
    new_sol = np.zeros(n)
    try:
        for i in range(n):
            new_sol[i] = x[i].X
    except:
        if(obj_type == -1):
            for i in range(n):
                new_sol[i] = lower_bound[i]
        else:
            for i in range(n):
                new_sol[i] = upper_bound[i]
        
    return new_sol

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type, now_sol, now_col):
    """
    Function Description:
    Solve the given problem instance using the Gurobi solver.

    Parameter Description:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable is the j-th variable in the i-th constraint.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 indicates <= and 2 indicates >=.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Indicates whether the problem is a maximization or minimization problem.
    """

    begin_time = time.time()
    model = Model("Gurobi")
    site_to_new = {}
    new_to_site = {}
    new_num = 0
    x = []
    for i in range(n):
        if(now_col[i] == 1):
            site_to_new[i] = new_num
            new_to_site[new_num] = i
            new_num += 1
            if(value_type[i] == 'B'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
            elif(value_type[i] == 'C'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
            else:
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
    
    coeff = 0
    for i in range(n):
        if(now_col[i] == 1):
            coeff += x[site_to_new[i]] * coefficient[i]
        else:
            coeff += now_sol[i] * coefficient[i]
    if(obj_type == -1):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    for i in range(m):
        constr = 0
        flag = 0
        for j in range(k[i]):
            if(now_col[site[i][j]] == 1):
                constr += x[site_to_new[site[i][j]]] * value[i][j]
                flag = 1
            else:
                constr += now_sol[site[i][j]] * value[i][j]

        if(flag == 1):
            if(constraint_type[i] == 1):
                model.addConstr(constr <= constraint[i])
            elif(constraint_type[i] == 2):
                model.addConstr(constr >= constraint[i])
            else:
                model.addConstr(constr == constraint[i])
        else:
            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    print(now_col)
            else:
                if(constr < constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    print(now_col)
    #model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    model.optimize()
    try:
        new_sol = np.zeros(n)
        for i in range(n):
            if(now_col[i] == 0):
                new_sol[i] = now_sol[i]
            else:
                if(value_type[i] == 'C'):
                    new_sol[i] = x[site_to_new[i]].X
                else:
                    new_sol[i] = (int)(x[site_to_new[i]].X)
            
        return new_sol, model.ObjVal, 1
    except:
        return -1, -1, -1

def eval(n, coefficient, new_sol):
    ans = 0
    for i in range(n):
        ans += coefficient[i] * new_sol[i]
    return(ans)



def select_neighborhood(n, current_solution, lp_solution):
    neighbor_score = np.zeros(n)
    for var_index in range(n):
        neighbor_score[var_index] = -abs(current_solution[var_index] - lp_solution[var_index])
    return neighbor_score

def greedy_one(now_instance_data, time_limit):
    begin_time = time.time()
    set_time = time_limit
    epsilon = 1e-3
    n = now_instance_data[0]
    m = now_instance_data[1]
    k = now_instance_data[2]
    site = now_instance_data[3]
    value = now_instance_data[4] 
    constraint = now_instance_data[5]
    constraint_type = now_instance_data[6] 
    coefficient = now_instance_data[7]
    obj_type = now_instance_data[8]
    lower_bound = now_instance_data[9]
    upper_bound = now_instance_data[10]
    value_type = now_instance_data[11]
    initial_sol = now_instance_data[12]

    choose = 0.5
    best_val = eval(n, coefficient, initial_sol)
    
    turn_time = [time.time() - begin_time]
    turn_ans = [best_val]

    #Find LP solution
    LP_sol = initial_LP_solution(n, m, k, site, value, constraint, constraint_type, coefficient, set_time * 0.3, obj_type, lower_bound, upper_bound, value_type)
    
    turn_limit = 100
    
    now_sol = initial_sol
    while(time.time() - begin_time <= set_time):
        #print("before", parts, time.time() - begin_time)
        #"n", "m", "k", "site", "value", "constraint", "initial_solution", "current_solution", "objective_coefficient"
        neighbor_score = select_neighborhood(
                            n, 
                            copy.deepcopy(now_sol), 
                            copy.deepcopy(LP_sol)
                        )
        #print("after", parts, time.time() - begin_time)
        indices = np.argsort(neighbor_score)[::-1]
        color = np.zeros(n)
        for i in range(int(n * choose)):
            color[indices[i]] = 1
        new_sol, now_val, now_flag = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(set_time - (time.time() - begin_time), turn_limit), obj_type, lower_bound, upper_bound, value_type, now_sol, color)
        if(now_flag == -1):
            continue
        
        #Maximize
        if(obj_type == -1):
            if(now_val > best_val):
                now_sol = new_sol
                best_val = now_val
        else:
            if(now_val < best_val):
                now_sol = new_sol
                best_val = now_val

        turn_ans.append(best_val) 
        turn_time.append(time.time() - begin_time)
    return(turn_ans, turn_time)

def split_problem(lp_file):
    """
    Function Description:
    Solve the given problem instance using the Gurobi solver.

    Parameter Description:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable is the j-th variable in the i-th constraint.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 indicates <= and 2 indicates >=.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Indicates whether the problem is a maximization or minimization problem.
    """
    model = gp.read(lp_file)
    n = model.NumVars
    m = model.NumConstrs
    k = []
    site = []
    value = []
    constraint = []
    constraint_type = []
    coefficient = []
    obj_type = model.ModelSense
    upper_bound = []
    lower_bound = []
    value_type = []
    var_name_to_index = {}

    objective = model.getObjective()
    temp_coeff = []
    temp_varname = []
    for i in range(objective.size()):
        temp_coeff.append(objective.getCoeff(i))
        temp_varname.append(objective.getVar(i).VarName)

    i = 0
    for var in model.getVars():
        var_name_to_index[var.VarName] = i
        upper_bound.append(var.UB)
        lower_bound.append(var.LB)
        value_type.append(var.VType)
        if var.VarName not in temp_varname:
            coefficient.append(0)
        else:
            coefficient.append(temp_coeff[temp_varname.index(var.VarName)])
        i+=1

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
            now_site.append(var_name_to_index[row.getVar(i).VarName])
            now_value.append(row.getCoeff(i))
        site.append(now_site)
        value.append(now_value)
        
    return n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type
