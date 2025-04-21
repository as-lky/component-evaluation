from __future__ import division
from __future__ import print_function

import os
import re
import glob
import time
import pickle
import random
import argparse
import numpy as np
import gurobipy as gp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from EGAT_models import SpGAT

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # List in tensor data format

    def forward(self, preds, labels):
        """
        preds: logits output values
        labels: labels
        """
        preds = F.softmax(preds, dim=1).to(device)
        eps = 1e-7
        target = self.one_hot(preds.size(1), labels).to(device)
        ce = (-1 * torch.log(preds + eps) * target).to(device)
        floss = (torch.pow((1 - preds), self.gamma) * ce).to(device)
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one



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


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')

parser.add_argument("--train_data_dir", type=str, help="the train instances input folder")
parser.add_argument("--model_save_dir", type=str, help="the model output directory")
parser.add_argument("--log_dir", type=str, help="the train tmp file restore") 
parser.add_argument("--random_feature", action='store_true', help="whether use random feature or not")
    
parser.add_argument('--seed', type=int, default=16, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=6, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

data_num = 20
data_features = []
data_labels = []
data_solution = []
data_edge_features = []
data_edge_A = []
data_edge_num_A = []
data_edge_B = []
data_edge_num_B = []
data_idx_train = []

def c(a):
    tmp = os.path.basename(a)
    tmp = re.match(r".*_([0-9]+)", tmp)
    tmp = tmp.group(1)
    return int(tmp) <= 9


log_dir = args.log_dir
train_data_dir = args.train_data_dir
random_feature = args.random_feature
model_save_dir = args.model_save_dir

train_data_path = train_data_dir
# load samples from data_path and divide them
DIR_BG = train_data_path + 'LP'
DIR_SOL = train_data_path + 'Pickle'

sample_names = os.listdir(DIR_BG)
sample_files = [ (os.path.join(DIR_BG,name), os.path.join(DIR_SOL,name).replace('lp','pickle')) for name in sample_names if not c(name)]

for ____ in sample_files:
    _ = ____[0]
    solution_path = ____[1]
    
    instance_name = os.path.basename(_)
    instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
    instance_name = instance_name.group(1)
    pk = os.path.join(log_dir, instance_name) + '.pickle'
    if not os.path.exists(pk):
        constraint_features, edge_indices, edge_features, variable_features, num_to_value, n = get_a_new2(_, random_feature)
        with open(solution_path, "rb") as f:
            solution = pickle.load(f)[0]
        sol = []
        for i in range(n):
            sol.append(solution[num_to_value[i]])
        with open(pk, "wb") as f:
            pickle.dump([variable_features, constraint_features, edge_indices, edge_features, sol], f)

    with open(pk, "rb") as f:
        problem = pickle.load(f)

    variable_features = problem[0]
    constraint_features = problem[1]
    edge_indices = problem[2]
    edge_feature = problem[3]
    optimal_solution = problem[4]

    #change
    n = len(variable_features)
    var_size = len(variable_features[0])
    m = len(constraint_features)
    con_size = len(constraint_features[0])

    edge_num = len(edge_indices[0])
    data_edge_num_A.append(edge_num)
    edge_num = len(edge_indices[0])
    data_edge_num_B.append(edge_num)

    edgeA = []
    edgeB = []
    edge_features = []
    for i in range(edge_num):
        edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
        edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])
        edge_features.append(edge_feature[i])
    edgeA = torch.as_tensor(edgeA)
    data_edge_A.append(edgeA)

    edgeB = torch.as_tensor(edgeB)
    data_edge_B.append(edgeB)

    edge_features = torch.as_tensor(edge_features)
    data_edge_features.append(edge_features)

    for i in range(m):
        for j in range(var_size - con_size):
            constraint_features[i].append(0)
    features = variable_features + constraint_features
    features = torch.as_tensor(features).float()
    data_features.append(features)

    #labelA = torch.tensor(patition_color)
    new_optimal_solution = []
    for item in optimal_solution:
        new_optimal_solution.append((int)(item))
    optimal_solution = new_optimal_solution
    num_label = [1, 1]
    num_label = torch.as_tensor(num_label).to(device)
    data_labels.append(num_label)

    for i in range(m):
        optimal_solution.append(0)

    labels = []
    #For Binary
    for i in range(n + m):
        if(optimal_solution[i] == 0):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    labels = torch.tensor(labels)

    labels = torch.as_tensor(optimal_solution)

    data_solution.append(labels)

    idx_train = torch.as_tensor(range(n))
    data_idx_train.append(idx_train)

# Model and optimizer
model = SpGAT(nfeat=data_features[0].shape[1],    # Feature dimension
            nhid=args.hidden,             # Feature dimension of each hidden layer
            nclass=1, # Number of classes
            dropout=args.dropout,         # Dropout
            nheads=args.nb_heads,         # Number of heads
            alpha=args.alpha)             # LeakyReLU alpha coefficient

optimizer = optim.Adam(model.parameters(),    
                       lr=args.lr,                        # Learning rate
                       weight_decay=args.weight_decay)    # Weight decay to prevent overfitting

if args.cuda: # Move to GPU
    model.to(device)
    for now_data in range(data_num):
        data_features[now_data] = data_features[now_data].to(device)
        data_labels[now_data] = data_labels[now_data].to(device)
        data_solution[now_data] = data_solution[now_data].to(device)
        data_edge_A[now_data] = data_edge_A[now_data].to(device)
        data_edge_B[now_data] = data_edge_B[now_data].to(device)
        data_edge_features[now_data] = data_edge_features[now_data].to(device)
        data_idx_train[now_data] = data_idx_train[now_data].to(device)


for now_data in range(data_num):
    data_features[now_data] = Variable(data_features[now_data])
    data_edge_A[now_data] = Variable(data_edge_A[now_data])
    data_edge_B[now_data] = Variable(data_edge_B[now_data])
    data_solution[now_data] = Variable(data_solution[now_data])
    # Define computation graph for automatic differentiation

def train(epoch, num):
    global data_edge_features
    t = time.time()

    output, select, data_edge_features[num] = model(data_features[num], data_edge_A[num], data_edge_B[num], data_edge_features[num].detach())
#    print(data_solution[num][idx_train])

    #lf = Focal_Loss(torch.as_tensor(data_labels[num]))
    #loss_train = lf(output[idx_train], data_solution[num][idx_train])

    #return loss_train
    choose = {}
    n = output.shape[0]
    for i in range(n):
        if(select[i] >= 0.5):
            choose[i] = 0
        else:
            choose[i] = 1
    new_idx_train = []
    for i in range(n):
        if(choose[i]):
            new_idx_train.append(i)
    
    set_c = 0.7
    if(len(new_idx_train) < set_c * n):
        loss_select = (set_c - len(new_idx_train) / n) ** 2
    else:
        loss_select = 0
    loss_func = torch.nn.MSELoss()
    loss = loss_func(output[new_idx_train], torch.tensor(np.array(optimal_solution))[new_idx_train].float()) + loss_select

    return loss



t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    now_loss = 0
    for i in range(5):
        now_data = random.randint(0, data_num - 1)
        now_loss += train(epoch, now_data)
    loss_values.append(now_loss)
    now_loss.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(now_loss))

#    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        torch.save(model.state_dict(), model_save_dir + 'model_best.pkl')
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:  # Stop if there's no improvement for several consecutive rounds
        break

#    files = glob.glob('*.pkl')
#    for file in files:
#        epoch_nb = int(file.split('.')[0])
#        if epoch_nb < best_epoch:
#            os.remove(file)

#files = glob.glob('*.pkl')
#for file in files:
#    epoch_nb = int(file.split('.')[0])
#    if epoch_nb > best_epoch:
#        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
#model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

print(loss_values)

