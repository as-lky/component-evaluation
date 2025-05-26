import gurobipy as gp

def split_problem(lp_file):
    '''
    Pass in an LP file and split the given problem
    '''
    model = gp.read(lp_file)
    value_to_num = {}
    value_num = 0
    #n represents the num of decision variables
    #m represents the num of constrains
    #k[i] represents the number of decision variables in the i-th constraint
    #site[i][j] represents which decision variable is the j-th decision variable in the i-th constraint
    #value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint
    #constraint[i] represents the number on the right-hand side of the i-th constraint
    #constraint_type[i] represents the type of the i-th constraint, where 1 indicates '<', 2 indicates '>', and 3 indicates '='
    #coefficient[i] represents the coefficient of the i-th decision variable in the objective function
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
            value_num += 1
        coefficient[value_to_num[val.VarName]] = val.Obj
        lower_bound[value_to_num[val.VarName]] = val.LB
        upper_bound[value_to_num[val.VarName]] = val.UB
        value_type[value_to_num[val.VarName]] = val.Vtype

    #1 for maximize，-1 for minimize
    obj_type = model.ModelSense
    if(obj_type == -1):
        obj_type = 'maximize'
    else:
        obj_type = 'minimize'
    return n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num