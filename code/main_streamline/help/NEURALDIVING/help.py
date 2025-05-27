import pickle
import torch
import torch_geometric
import gurobipy as gp
import random
from gurobipy import *
import time

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, now_col, lower_bound, upper_bound, value_type):
    '''
    Function Description:
    Use Gurobi solver to solve the problem based on the provided problem instance and current solution and current selection.

    Parameter description:
    -N: The number of decision variables in the problem instance.
    -M: The number of constraints for problem instances.
    -K: k [i] represents the number of decision variables for the i-th constraint.
    -Site: site [i] [j] represents which decision variable is the jth decision variable of the i-th constraint.
    -Value: value [i] [j] represents the coefficient of the jth decision variable of the i-th constraint.
    -Constraint: constraint [i] represents the number to the right of the i-th constraint.
    -Constrict_type: constrict_type [i] represents the type of the i-th constraint, 1 represents<=, 2 represents>=
    -Coefficient: coefficient [i] represents the coefficient of the i-th decision variable in the objective function.
    -Time_imit: Maximum solution time.
    -Obj_type: Is the problem a maximization problem or a minimization problem.
    -Now_sol: represents the current solution.
    -Now_col: represents the current selection of decision variables, 0 means selected, 1 means not selected.
    '''
    begin_time = time.time()
    model = Model("Gurobi")
    model.feasRelaxS(0,False,False,True)
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
                    # No feasible solution
                    print("QwQ fine")
                    print(constr,  constraint[i])
                    return -1, -1, -1, -1
            else:
                if(constr < constraint[i]):
                    print("QwQ fine")
                    print(constr,  constraint[i])
                    return -1, -1, -1, -1
    
    coeff = 0
    flag = 0
    for i in range(n):
        if(now_col[i] == 1):
            coeff += x[site_to_new[i]] * coefficient[i]
            flag = 1
        else:
            coeff += now_sol[i] * coefficient[i]
    
    if flag == 1:
        if(obj_type == 'maximize'):
            model.setObjective(coeff, GRB.MAXIMIZE)
        else:
            model.setObjective(coeff, GRB.MINIMIZE)
                    
        model.setParam('SolutionLimit', 1)
        model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
        model.optimize()

    try:
        new_sol = []
        for i in range(n):
            if(now_col[i] == 0):
                new_sol.append(now_sol[i])
            else:
                if(value_type[i] == 'C'):
                    new_sol.append(x[site_to_new[i]].X)
                else:
                    new_sol.append((int)(x[site_to_new[i]].X))
        if model.NumVars == 0:
            return 1, new_sol, coeff, 0
        
        return 1, new_sol, model.ObjVal, model.MIPGap
    except:
        return -1, -1, -1, -1

# bipartite graph data
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.assignment = assignment

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# tripartite graph data
class TripartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node tripartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        obj_features,
        obj_variable_val,
        obj_constraint_val,
        edge_obj_var,
        edge_obj_con,
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.obj_features = obj_features
        self.obj_variable_val = obj_variable_val
        self.obj_constraint_val = obj_constraint_val
        self.edge_obj_var = edge_obj_var
        self.edge_obj_con = edge_obj_con
        self.assignment = assignment

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "edge_obj_var":
            return torch.tensor(
                [[1], [self.variable_features.size(0)]]
            )
        elif key == "edge_obj_con":
            return torch.tensor(
                [[1], [self.constraint_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# graph dataset
class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, tripartite=False):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.tripartite = tripartite

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        if not self.tripartite:
            with open(self.sample_files[index], "rb") as f:
                [variable_features, constraint_features, edge_indices, edge_features, solution] = pickle.load(f)

            graph = BipartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(solution)
            )

            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = len(constraint_features) + len(variable_features)
            graph.cons_nodes = len(constraint_features)
            graph.vars_nodes = len(variable_features)

            return graph
        else:
            with open(self.sample_files[index], "rb") as f:
                [variable_features, constraint_features, edge_indices, edge_features, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con, solution] = pickle.load(f)    

            graph = TripartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(obj_features),
                torch.FloatTensor(obj_variable_val),
                torch.FloatTensor(obj_constraint_val),
                torch.LongTensor(edge_obj_var),
                torch.LongTensor(edge_obj_con),
                torch.FloatTensor(solution)
            )

            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = len(constraint_features) + len(variable_features) + 1
            graph.cons_nodes = len(constraint_features)
            graph.vars_nodes = len(variable_features)
            graph.obj_nodes = 1

            return graph
        
# function to read the *.lp file and return the basic information of the instance and the encoded bipartite graph features
# random_feature is true when encoding the instance into a graph with random features
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

    return constraint_features, edge_indices, edge_features, variable_features, num_to_value, n

# function to read the *.lp file and return the basic information of the instance and the encoded tripartite graph features
# random_feature is true when encoding the instance into a graph with random features
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

    return constraint_features, edge_indices, edge_features, variable_features, num_to_value, n, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con
