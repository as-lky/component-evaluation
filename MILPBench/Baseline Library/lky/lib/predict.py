from http.client import GATEWAY_TIMEOUT
import torch
import os
import re
import gurobipy as gp
import pyscipopt
import subprocess
#import cplex
from typing import Type, cast, Self
from .mod import Component, Graphencode2Predict, Predict2Modify, Cantsol, Cansol2M
from .help.NEURALDIVING.test import GraphDataset
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
            cls=  GAT
        else:
            raise ValueError("Predict component type is not defined")
        
        return super().__new__( cast(type[Self], cls) )

    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)


    def work(self, input: Graphencode2Predict) -> Predict2Modify:...

class Gurobi(Predict):
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
