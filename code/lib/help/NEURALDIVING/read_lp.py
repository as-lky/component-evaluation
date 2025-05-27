from gurobipy import *
from pathlib import Path
import numpy as np
import argparse
import pickle
import random
import time
import gurobipy as gp
import os


def get_a_new2(instance, random_feature = False):
    model = gp.read(instance)
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
    
    
    variable_features = []
    constraint_features = []
    edge_indices = [[], []] 
    edge_features = []

    for i in range(n):
        now_variable_features = []
        now_variable_features.append(coefficient[i])
        if(lower_bound[i] == float("-inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(lower_bound[i])
        if(upper_bound[i] == float("inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(upper_bound[i])
        if(value_type[i] == 'C'):
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
        if random_feature:
            now_variable_features.append(random.random())
        variable_features.append(now_variable_features)
    
    for i in range(m):
        now_constraint_features = []
        now_constraint_features.append(constraint[i])
        now_constraint_features.append(constraint_type[i])
        if random_feature:
            now_constraint_features.append(random.random())
        constraint_features.append(now_constraint_features)
    
    for i in range(m):
        for j in range(k[i]):
            edge_indices[0].append(i)
            edge_indices[1].append(site[i][j])
            edge_features.append([value[i][j]])

    return constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value


def get_a_new3(instance, random_feature = False):
    model = gp.read(instance)
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
    
    
    edge_obj_var = [[0] * n, [i for i in range(n)]]
    edge_obj_con = [[0] * m, [i for i in range(m)]]
    obj_variable_val = []
    obj_constraint_val = []
    obj_features = [[]]
    
    variable_features = []
    constraint_features = []
    edge_indices = [[], []] 
    edge_features = []

    cnt = 0
    MAX, MIN = -2e9, 2e9
    for i in range(n):
        obj_variable_val.append([coefficient[i]])
        if coefficient[i] != 0:
            cnt += 1
        MAX = max(MAX, coefficient[i])
        MIN = min(MIN, coefficient[i])
    obj_features[0] = [obj_type, cnt, MAX, MIN]
    if random_feature:
        obj_features[0].append(random.random())
    for i in range(m):
        obj_constraint_val.append([constraint[i]])
        
 #   tmp_coefficient = [abs(_) for _ in coefficient if _ != 0]
 #   tmp_coefficient = sorted(tmp_coefficient)
 #   threshold = tmp_coefficient[int(len(tmp_coefficient) * 0.5)]

    # for i in range(n):
    #     tmp = []
    #     coef = coefficient[i]
    #     if coef != 0:
    #         tmp.append(obj_type)
    #         tmp.append(1 if coef > 0 else -1)
    #         tmp.append(1 if abs(coef) > threshold else 0)
    #         if random_feature:
    #             tmp.append(random.random())
    #         obj_features.append(tmp)
    #         obj_num.append(i)
            
            
    # obj_features[0] = [cnt, obj_type]
    # if random_feature:
    #     obj_features[0].append(random.random())
    
    
    for i in range(n):
        now_variable_features = []
        now_variable_features.append(coefficient[i])
        if(lower_bound[i] == float("-inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(lower_bound[i])
        if(upper_bound[i] == float("inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(upper_bound[i])
        if(value_type[i] == 'C'):
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
        if random_feature:
            now_variable_features.append(random.random())
        variable_features.append(now_variable_features)
    
    for i in range(m):
        now_constraint_features = []
        now_constraint_features.append(constraint[i])
        now_constraint_features.append(constraint_type[i])
        if random_feature:
            now_constraint_features.append(random.random())
        constraint_features.append(now_constraint_features)
    
    for i in range(m):
        for j in range(k[i]):
            edge_indices[0].append(i)
            edge_indices[1].append(site[i][j])
            edge_features.append([value[i][j]])

    return constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con


def optimize(
    mode : str,
    path : str,
):
    '''
    Function Description:
    Based on the input parameters, call the specified algorithm and solver to optimize the optimization problem in datapickle in the same directory.

    Parameter description:
    -Number: integer type, representing the number of instances generated.
    -Suboptimal: integer type, representing the number of suboptimal solutions generated.
    '''
    path = f'instances/{path}'
    if(mode == "train"):
        lp_path = path + "/LP"
        pickle_path = path + "/Pickle"
    else:
        lp_path = path + "/LP_test"
        pickle_path = path + "/Pickle"
    
    lp_files = []
    lp_name = []
    for filename in os.listdir(lp_path):
        if filename.endswith('.lp'):
            file_path = os.path.join(lp_path, filename)
            lp_files.append(file_path)
            lp_name.append(filename)
    #print(lp_name)
    if not os.path.exists(path + "/" + mode):
        os.mkdir(path + "/" + mode)

    for num in range(len(lp_name)):
        model = read(lp_files[num])
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
        print(n, m)
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
        
        
        variable_features = []
        constraint_features = []
        edge_indices = [[], []] 
        edge_features = []

        for i in range(n):
            now_variable_features = []
            now_variable_features.append(coefficient[i])
            if(lower_bound[i] == float("-inf")):
                now_variable_features.append(0)
                now_variable_features.append(0)
            else:
                now_variable_features.append(1)
                now_variable_features.append(lower_bound[i])
            if(upper_bound[i] == float("inf")):
                now_variable_features.append(0)
                now_variable_features.append(0)
            else:
                now_variable_features.append(1)
                now_variable_features.append(upper_bound[i])
            if(value_type[i] == 'C'):
                now_variable_features.append(0)
            else:
                now_variable_features.append(1)
            now_variable_features.append(random.random())
            variable_features.append(now_variable_features)
        
        for i in range(m):
            now_constraint_features = []
            now_constraint_features.append(constraint[i])
            now_constraint_features.append(constraint_type[i])
            now_constraint_features.append(random.random())
            constraint_features.append(now_constraint_features)
        
        for i in range(m):
            for j in range(k[i]):
                edge_indices[0].append(i)
                edge_indices[1].append(site[i][j])
                edge_features.append([value[i][j]])
        
        pickle_file = pickle_path + "/" + lp_name[num][:-3] + ".pickle"
        with open(pickle_file, 'rb')as f:
            x_dict, gap = pickle.load(f)

        optimal_solution = []
        for i in range(n):
            optimal_solution.append(x_dict[num_to_value[i]])

        with open(path + "/" + mode + '/pair' + str(num) + '.pickle', 'wb') as f:
            pickle.dump([variable_features, constraint_features, edge_indices, edge_features, optimal_solution], f)
        with open(path + "/" + mode + '/data' + str(num) + '.pickle', 'wb') as f:
            pickle.dump([obj_type, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, optimal_solution], f)
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type = str, default = "train", help = 'Running mode.')
    parser.add_argument('path', type = str, default = "", help = 'Running mode.')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    optimize(**vars(args))
