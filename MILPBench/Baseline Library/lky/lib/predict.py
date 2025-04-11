from http.client import GATEWAY_TIMEOUT
import torch
import os
import re
import gurobipy as gp
import numpy as np
import pyscipopt
import subprocess
import pickle
import random
#import cplex
from typing import Type, cast, Self
from .mod import Component, Graphencode2Predict, Predict2Modify, Cantsol, Cansol2M
from .help.NEURALDIVING.test import GraphDataset
from .help.NEURALDIVING.help import get_a_new2 as get_a_new2_gcn
from .help.LIGHT.help import get_a_new2 as get_a_new2_gat

from .help.LIGHT.EGAT_models import SpGAT
import torch_geometric

class Predict(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "gcn":
            cls = GCN
        elif component == "gurobi":
            cls = Gurobi 
        elif component == "scip":
            cls = SCIP
        elif component == "cplex":
            cls = CPLEX
        elif component == "l2bs":
            cls = L2BS
        elif component == 'gat':
            cls = GAT
        else:
            raise ValueError("Predict component type is not defined")
        
        return super().__new__( cast(type[Self], cls) )

    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)


    def work(self, input: Graphencode2Predict) -> Predict2Modify:...

class Gurobi(Predict): # TODO: 找到解立刻停止
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
        
        cansol = {}
        
        model = gp.read(self.instance)
        model.setParam('TimeLimit', self.time_limit)
        model.optimize()
        for var in model.getVars():
            cansol[var.VarName] = var.X
        
        self.end()
        return Cansol2M(model.ObjVal, cansol, model.MIPGap)
    
class SCIP(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
    
        solver = pyscipopt.Model()
        solver.readProblem(self.instance)
        solver.setRealParam('limits/time', self.time_limit)
        solver.optimize()
    
        cansol = {}
        
        for var in solver.getVars():
            cansol[var.name] = solver.getVal(var)

        self.end()
        return Cansol2M(solver.getObjVal(), cansol, solver.getGap())
    
class CPLEX(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
        
        cansol = {}
        
        model = cplex.Cplex()
        model.read(self.instance)
        model.parameters.timelimit.set(self.time_limit)
        model.solve()
        
        for var_name, var_value in zip(model.variables.get_names(), model.solution.get_values()):
            cansol[var_name] = var_value

        self.end()
        return Cansol2M(model.solution.get_objective_value(), cansol, model.solution.MIP.get_mip_relative_gap())


# class GCN(Predict):
        
#     def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
#         super().__init__(component, device, taskname, instance, sequence_name)
#         if "train_data_dir" in kwargs:
#             self.train_data_dir = kwargs["train_data_dir"]
#         ... # tackle parameters 
 
#     def work(self, input: Graphencode2Predict) -> Cantsol:    
        
#         self.begin()
#         # first check the model, if there is not then train using train instances
#         if self.taskname == "IP":
#             #Add position embedding for IP model, due to the strong symmetry
#             from .help.GCN.GCN import GNNPolicy_position as GNNPolicy
#         else:
#             from .help.GCN.GCN import GNNPolicy
        
#         DEVICE = self.device     
        
#         instance_name = os.path.basename(self.instance)
#         instance_name = re.match(r"(.*)_[0-9]+", instance_name)
#         if instance_name == None:
#             raise ValueError("instance name error!")
#         else :
#             instance_name = instance_name.group(1)
            
#         pathstr = ""
#         # 模型训练不需要以sequence_name做路径 因为其与其他部分无关 只保留instance_name 和参数可以确保可复用性 ×
#         model_dir = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
#         model_path = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/model_best.pth'
#         if os.path.exists(model_path): # TODO : add parameter for model name
#             pathstr = model_path
#         else :
#             if not os.path.isdir('./logs/'):
#                 os.mkdir('./logs')
#             if not os.path.isdir(f'./logs/train/'):
#                 os.mkdir('./logs/train')
#             if not os.path.isdir(f'./logs/train/{self.taskname}/'):
#                 os.mkdir(f'./logs/train/{self.taskname}')
#             if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/'):
#                 os.mkdir(f'./logs/train/{self.taskname}/{instance_name}')
#             if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
#                 os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
#             if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'):
#                 os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
#             if not os.path.isdir('./Model/'):
#                 os.mkdir('./Model/')
#             if not os.path.isdir(f'./Model/{self.taskname}'):
#                 os.mkdir(f'./Model/{self.taskname}')
#             if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}'):
#                 os.mkdir(f'./Model/{self.taskname}/{instance_name}')
#             if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
#                 os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
#             if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}'):
#                 os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
#             W = f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
#             if self.sequence_name[0][-1] == 'r':
#                 subprocess.run(["python", "lib/help/GCN/trainPredictModel.py", "--device", f"{self.device}", "--taskname", f"{self.taskname}", "--train_data_dir", f"{self.train_data_dir}",
#                                 "--log_dir", f"{W}", "--model_save_dir", f"{model_dir}", "--random_feature"])    
#             else:
#                 subprocess.run(["python", "lib/help/GCN/trainPredictModel.py", "--device", f"{self.device}", "--taskname", f"{self.taskname}", "--train_data_dir", f"{self.train_data_dir}",
#                                 "--log_dir", f"{W}", "--model_save_dir", f"{model_dir}"])    
#             pathstr = model_path
#             # train_data_dir + LP / Pickle    
            
#         policy = GNNPolicy(random_feature=True if self.sequence_name[0][-1] == 'r' else False).to(DEVICE)
#         state = torch.load(pathstr, map_location=torch.device(DEVICE)) # TODO: check why cuda?
#         policy.load_state_dict(state)
        
#         BD = policy(
#             input.constraint_features.to(DEVICE),
#             input.edge_indices.to(DEVICE),
#             input.edge_features.to(DEVICE),
#             input.variable_features.to(DEVICE),
#         ).sigmoid().cpu().squeeze()

#         # align the variable name betweend the output and the solver
#         all_varname=[]
#         for name in input.v_map:
#             all_varname.append(name)
#         scores=[] # get a list of (index, VariableName, Prob)
#         for i in range(len(input.v_map)):
#             scores.append([i, all_varname[i], BD[i].item()])
#         self.end()
#         return Cantsol(scores)

class GCN(Predict):
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        if "train_data_dir" in kwargs:
            self.train_data_dir = kwargs["train_data_dir"]
        else :
            self.train_data_dir = None
        ... # tackle parameters 
 
    def work(self, input: Graphencode2Predict) -> Cantsol:    
        
        self.begin()
        # first check the model, if there is not then train using train instances
        from .help.NEURALDIVING.graphcnn import GNNPolicy
        
        DEVICE = self.device     
        
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*)_[0-9]+", instance_name)
        if instance_name == None:
            raise ValueError("instance name error!")
        else :
            instance_name = instance_name.group(1)
            
        pathstr = ""
        # 模型训练不需要以sequence_name做路径 因为其与其他部分无关 只保留instance_name 和参数可以确保可复用性 ×
        model_dir = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
        model_path = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/model_best.pkl'
        W = f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'

        if os.path.exists(model_path): # TODO : add parameter for model name
            pathstr = model_path
        else :
            if not os.path.isdir('./logs/'):
                os.mkdir('./logs')
            if not os.path.isdir(f'./logs/train/'):
                os.mkdir('./logs/train')
            if not os.path.isdir(f'./logs/train/{self.taskname}/'):
                os.mkdir(f'./logs/train/{self.taskname}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
            if not os.path.isdir('./Model/'):
                os.mkdir('./Model/')
            if not os.path.isdir(f'./Model/{self.taskname}'):
                os.mkdir(f'./Model/{self.taskname}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
#            if self.sequence_name[0][-1] == 'r':
#                subprocess.run(["python", "lib/help/GCN/trainPredictModel.py", "--device", f"{self.device}", "--taskname", f"{self.taskname}", "--train_data_dir", f"{self.train_data_dir}",
#                                "--log_dir", f"{W}", "--model_save_dir", f"{model_dir}", "--random_feature"])    
#            else:
#                subprocess.run(["python", "lib/help/GCN/trainPredictModel.py", "--device", f"{self.device}", "--taskname", f"{self.taskname}", "--train_data_dir", f"{self.train_data_dir}",
#                                "--log_dir", f"{W}", "--model_save_dir", f"{model_dir}"])    
 
 
            if self.sequence_name[0][-1] == 'r':
                subprocess.run(["python", "lib/help/NEURALDIVING/train.py", "--train_data_dir", f"{self.train_data_dir}",
                                    "--model_save_dir", f"{model_dir}", "--log_dir", f"{W}", "--random_feature"])    
            else:
                subprocess.run(["python", "lib/help/NEURALDIVING/train.py", "--train_data_dir", f"{self.train_data_dir}",
                                    "--model_save_dir", f"{model_dir}", "--log_dir", f"{W}"])    
 
            pathstr = model_path
            # train_data_dir + LP / Pickle    
            
        policy = GNNPolicy(random_feature=True if self.sequence_name[0][-1] == 'r' else False).to(DEVICE)
        policy.load_state_dict(torch.load(model_path, policy.state_dict()))
        
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
        instance_name = instance_name.group(1)
        pk = os.path.join(W, instance_name) + '.pickle'
    
        if not os.path.exists(pk):
            constraint_features, edge_indices, edge_features, variable_features, num_to_value, n = get_a_new2_gcn(self.instance, random_feature=True if self.sequence_name[0][-1] == 'r' else False)            
            sol = []
            with open(pk, "wb") as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features, sol], f)
        
        file = [pk]
        data = GraphDataset(file)
        loader = torch_geometric.loader.DataLoader(data, batch_size = 1)

        logits, select = None, None
        for batch in loader:
            batch = batch.to(self.device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits, select = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )

        self.end()
        return Cantsol(logits, select)


    
class L2BS(Predict):
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        if "train_data_dir" in kwargs:
            self.train_data_dir = kwargs["train_data_dir"]
        ... # tackle parameters 
 
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
        
        device = self.device
        problem = self.taskname
        instances = [self.instance]
        seeds = [0]
        gnn_models = ['supervised'] # Can be supervised

        instances += [{'type': f'{problem}', 'path': f'{self.instance}'}]

        time_limit = self.time_limit
        branching_policies = []

        # GNN models
        for model in gnn_models:
            for seed in seeds:
                branching_policies.append({
                    'type': 'gnn',
                    'name': model,
                    'seed': seed,
                })

        print(f"problem: {problem}")
        print(f"device: {self.device}")
        print(f"time limit: {time_limit} s")

        import torch
        from .help.LEARNBRANCH.model import GNNPolicy

        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*)_[0-9]+", instance_name)
        if instance_name == None:
            raise ValueError("instance name error!")
        else :
            instance_name = instance_name.group(1)

        model_path = f"./Model/{problem}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/train_params.pkl"



        # load and assign tensorflow models to policies (share models and update parameters)
        loaded_models = {}
        for policy in branching_policies:
            if policy['type'] == 'gnn':
                if policy['name'] not in loaded_models:
                    ### MODEL LOADING ###
                    model = GNNPolicy().to(device)
                    if policy['name'] == 'supervised':
                        model.load_state_dict(torch.load())
                    else:
                        raise Exception(f"Unrecognized GNN policy {policy['name']}")
                    loaded_models[policy['name']] = model

                policy['model'] = loaded_models[policy['name']]

        print("running SCIP...")

        fieldnames = [
            'policy',
            'seed',
            'type',
            'instance',
            'nnodes',
            'nlps',
            'stime',
            'gap',
            'status',
            'walltime',
            'proctime',
        ]
        os.makedirs('results', exist_ok=True)
        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': time_limit,
                        'timing/clocktype': 1, 'branching/vanillafullstrong/idempotent': True}

        with open(f"results/{result_file}", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for instance in instances:
                print(f"{instance['type']}: {instance['path']}...")

                for policy in branching_policies:
                    if policy['type'] == 'internal':
                        # Run SCIP's default brancher
                        env = ecole.environment.Configuring(scip_params={**scip_parameters,
                                                            f"branching/{policy['name']}/priority": 9999999})
                        env.seed(policy['seed'])

                        walltime = time.perf_counter()
                        proctime = time.process_time()

                        env.reset(instance['path'])
                        _, _, _, _, _ = env.step({})

                        walltime = time.perf_counter() - walltime
                        proctime = time.process_time() - proctime

                    elif policy['type'] == 'gnn':
                        # Run the GNN policy
                        env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                                        scip_params=scip_parameters)
                        env.seed(policy['seed'])
                        torch.manual_seed(policy['seed'])

                        walltime = time.perf_counter()
                        proctime = time.process_time()

                        observation, action_set, _, done, _ = env.reset(instance['path'])
                        while not done:
                            with torch.no_grad():
                                observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                                            torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                                            torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                                            torch.from_numpy(observation.variable_features.astype(np.float32)).to(device))

                                logits = policy['model'](*observation)
                                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                                observation, action_set, _, done, _ = env.step(action)

                        walltime = time.perf_counter() - walltime
                        proctime = time.process_time() - proctime

                    scip_model = env.model.as_pyscipopt()
                    stime = scip_model.getSolvingTime()
                    nnodes = scip_model.getNNodes()
                    nlps = scip_model.getNLPs()
                    gap = scip_model.getGap()
                    status = scip_model.getStatus()

                    pickle_folder = f'Pickle/{args.problem}/seed{args.seed}l2b_Pickle'
                    if not os.path.exists(pickle_folder):
                        os.makedirs(pickle_folder)

                    ans = {}
                    for var in scip_model.getVars():
                        ans[var.name] = scip_model.getVal(var)
                    with open(pickle_folder + '/' + (os.path.split(instance['path'])[1])[:-3] + '.pickle', 'wb')as f:
                        pickle.dump([ans, gap], f)

                    writer.writerow({
                    #     'policy': f"{policy['type']}:{policy['name']}",
                        'seed': policy['seed'],
                    #     'type': instance['type'],
                        'instance': instance['path'],
                        # 'nnodes': nnodes,
                        # 'nlps': nlps,
                        'stime': stime,
                        'gap': gap,
                        # 'status': status,
                        # 'walltime': walltime,
                        # 'proctime': proctime,
                    })
                    csvfile.flush()

                    print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")
        # TODO : eat shit!
        self.end()
        return Cantsol(scores)
    
class GAT(Predict):
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        if "train_data_dir" in kwargs:
            self.train_data_dir = kwargs["train_data_dir"]
        ... # tackle parameters 
 
    def work(self, input: Graphencode2Predict) -> Cantsol:    
        
        self.begin()
        
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*)_[0-9]+", instance_name)
        if instance_name == None:
            raise ValueError("instance name error!")
        else :
            instance_name = instance_name.group(1)
            
        pathstr = ""
        # 模型训练不需要以sequence_name做路径 因为其与其他部分无关 只保留instance_name 和参数可以确保可复用性 ×
        model_dir = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
        model_path = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/model_best.pkl'
        W = f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'

        if os.path.exists(model_path): # TODO : add parameter for model name
            pathstr = model_path
        else :
            if not os.path.isdir('./logs/'):
                os.mkdir('./logs')
            if not os.path.isdir(f'./logs/train/'):
                os.mkdir('./logs/train')
            if not os.path.isdir(f'./logs/train/{self.taskname}/'):
                os.mkdir(f'./logs/train/{self.taskname}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
            if not os.path.isdir('./Model/'):
                os.mkdir('./Model/')
            if not os.path.isdir(f'./Model/{self.taskname}'):
                os.mkdir(f'./Model/{self.taskname}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
            if self.sequence_name[0][-1] == 'r':
                subprocess.run(["python", "lib/help/LIGHT/train.py", "--train_data_dir", f"{self.train_data_dir}",
                                    "--model_save_dir", f"{model_dir}", "--log_dir", f"{W}", "--random_feature", "--no-cuda"])     #TODO: remove no-cuda
            else:
                subprocess.run(["python", "lib/help/LIGHT/train.py", "--train_data_dir", f"{self.train_data_dir}",
                                    "--model_save_dir", f"{model_dir}", "--log_dir", f"{W}", "--no-cuda"])
                
            pathstr = model_path
            # train_data_dir + LP / Pickle    
                
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
        instance_name = instance_name.group(1)
        
        pk_feature = os.path.join(W, instance_name) + '.pickle'
        if not os.path.exists(pk_feature):
            constraint_features, edge_indices, edge_features, variable_features, num_to_value, n = get_a_new2_gat(self.instance, random_feature=True if self.sequence_name[0][-1] == 'r' else False)
            sol = []
            with open(pk_feature, "wb") as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features, sol], f)

        with open(pk_feature, 'rb') as f:
            problem = pickle.load(f)
                
        variable_features = problem[0]
        constraint_features = problem[1]
        edge_indices = problem[2]
        edge_feature = problem[3]
        n = len(variable_features)
        var_size = len(variable_features[0])
        m = len(constraint_features)
        con_size = len(constraint_features[0])
        edge_num = len(edge_indices[0])

        edgeA = []
        edgeB = []
        edge_features = []
        for i in range(edge_num):
            edge_feature[i][0] /= n
        for i in range(edge_num):
            edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
            edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])
            edge_features.append(edge_feature[i])
        edgeA = torch.as_tensor(edgeA)
        edgeB = torch.as_tensor(edgeB)
        edge_features = torch.as_tensor(edge_features)

        for i in range(m):
            for j in range(var_size - con_size):
                constraint_features[i].append(0)
        features = variable_features + constraint_features
        features = torch.as_tensor(features)

        idx_test = torch.tensor(range(n))

        ##Predict
        #FENNEL
        partition_num = int(n / 20000)
        partition_var = []
        for i in range(partition_num):
            partition_var.append([])
        vertex_num = n + m
        edge_num = 0

        edge = []
        edge_val = []
        for i in range(vertex_num):
            edge.append([])
            edge_val.append([])
        for i in range(len(edgeA)):
            edge[edgeA[i][0]].append(edgeA[i][1])
            edge_val[edgeA[i][0]].append(1)
            edge[edgeA[i][1]].append(edgeA[i][0])
            edge_val[edgeA[i][1]].append(1)
            edge_num += 2

        alpha = (partition_num ** 0.5) * edge_num / (vertex_num ** (2 / 3))
        gamma = 1.5
        balance = 1.1

        visit = np.zeros(vertex_num, int)
        order = []
        for i in range(vertex_num):
            if(visit[i] == 0):
                q = []
                q.append(i)
                visit[i] = 1
                now = 0
                while(now < len(q)):
                    order.append(q[now])
                    for neighbor in edge[q[now]]:
                        if(visit[neighbor] == 0):
                            q.append(neighbor)
                            visit[neighbor] = 1
                    now += 1

        color = np.zeros(vertex_num, int)
        for i in range(vertex_num):
            color[i] = -1
        cluster_num = np.zeros(partition_num)
        score = np.zeros(partition_num, float)
        for i in range(vertex_num):
            now_vertex = order[i]
            load_limit = balance * vertex_num / partition_num
            for j in range(len(edge[now_vertex])):
                neighbor = edge[now_vertex][j]
                if(color[neighbor] != -1):
                    score[color[neighbor]] += edge_val[now_vertex][j]
            
            now_score = -2e9
            now_site = -1
            for j in range(len(edge[now_vertex])):
                neighbor = edge[now_vertex][j]
                if(color[neighbor] != -1):
                    if(score[color[neighbor]] > now_score):
                        now_score = score[color[neighbor]]
                        now_site = color[neighbor]
            neighbor = random.randint(0, partition_num - 1)
            if(score[neighbor] > now_score):
                now_score = score[neighbor]
                now_site = neighbor
            
            color[now_vertex] = now_site
            score[now_site] += alpha * gamma * (cluster_num[now_site] ** (gamma - 1))
            cluster_num[now_site] += 1
            score[now_site] -= alpha * gamma * (cluster_num[now_site] ** (gamma - 1))
            if(now_vertex < n):
                partition_var[now_site].append(now_vertex - n)

        color_site_to_num = []
        num_to_color_site = []
        color_site_num = []
        color_edgeA = []
        color_edgeB = []
        color_edge_features = []
        color_features = []
        color_edge_to_num = []
        for i in range(partition_num):
            color_site_to_num.append([])
            color_site_num.append(0)
            color_features.append([])
            color_edgeA.append([])
            color_edgeB.append([])
            color_edge_features.append([])
            color_edge_to_num.append([])

        for i in range(vertex_num):
            num_to_color_site.append(color_site_num[color[i]])
            color_site_num[color[i]] += 1
            color_site_to_num[color[i]].append(i)
            color_features[color[i]].append(features[i])

        edge_num = len(edge_indices[0])
        for i in range(edge_num):
            if(color[edge_indices[1][i]] == color[edge_indices[0][i] + n]):
                now_color = color[edge_indices[1][i]]
                color_edgeA[now_color].append([num_to_color_site[edge_indices[1][i]], num_to_color_site[edge_indices[0][i] + n]])
                color_edgeB[now_color].append([num_to_color_site[edge_indices[0][i] + n], num_to_color_site[edge_indices[1][i]]])
                color_edge_features[now_color].append(edge_feature[i])
                color_edge_to_num[now_color].append(i)

        path_model = model_path
        model = SpGAT(nfeat=features.shape[1],    # Feature dimension
                    nhid=64,                    # Feature dimension of each hidden layer
                    nclass=2,                   # Number of classes
                    dropout=0.5,                # Dropout
                    nheads=6,                   # Number of heads
                    alpha=0.2)                  # LeakyReLU alpha coefficient
        state_dict_load = torch.load(path_model)
        model.load_state_dict(state_dict_load)
        model.to(self.device)

        def compute_test(features, edgeA, edgeB, edge_features):
            model.eval()
            output, new_edge_feat = model(features, edgeA, edgeB, edge_features)
            #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            #acc_test = accuracy(output[idx_test], labels[idx_test])
            #print("Test set results:",
            #      "loss= {:.4f}".format(loss_test.data.item()))
            return(output, new_edge_feat)

        predict = []
        new_edge_feat = []
        for i in range(n + m):
            predict.append([])
        for i in range(edge_num):
            new_edge_feat.append(0)
        for i in range(partition_num):
            now_predict, now_new_edge_feat = compute_test(torch.tensor([item.cpu().detach().numpy() for item in color_features[i]]).cuda().float().to(device), torch.as_tensor(color_edgeA[i]).to(device), torch.as_tensor(color_edgeB[i]).to(device), torch.as_tensor(color_edge_features[i]).float().to(device))
            for j in range(len(color_site_to_num[i])):
                if(color_site_to_num[i][j] < n):
                    predict[color_site_to_num[i][j]] = now_predict[j].cpu().detach().numpy()
            for j in range(len(color_edge_to_num[i])):
                new_edge_feat[color_edge_to_num[i][j]] = now_new_edge_feat[j].cpu().detach().numpy()

        print(len(predict))
        
        self.end()
        return Cantsol(predict, predict)
