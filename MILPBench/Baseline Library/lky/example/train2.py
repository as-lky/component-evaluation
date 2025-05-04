from __future__ import division
from __future__ import print_function

import os
import glob
import time
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models2 import SpGAT

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss,self).__init__()
        self.gamma = gamma
        self.weight = weight        # 是tensor数据格式的列表

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        preds = F.softmax(preds,dim=1).to(device)
        #print(preds)
        eps = 1e-7

        target = self.one_hot(preds.size(1), labels).to(device)
        #print(target)

        ce = (-1 * torch.log(preds+eps) * target).to(device)
        #print(ce)
        floss = (torch.pow((1-preds), self.gamma) * ce).to(device)
        #floss = (torch.pow(torch.pow((1-preds), self.gamma), 1 / self.gamma) * ce).to(device)
        #print(floss)
        floss = torch.mul(floss, self.weight)
        #print(floss)
        floss = torch.sum(floss, dim=1)
        #print(floss)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0),num))
        one[range(labels.size(0)),labels] = 1
        return one


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=16, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=6, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load dat
data_num = 30
data_features = []
data_labels = []
data_solution = []
data_edge_features = []
data_edge_A = []
data_edge_num_A = []
data_edge_B = []
data_edge_num_B = []
data_idx_train = []
for now_data in range(data_num):
    if(os.path.exists('./example-IS-e/pair' + str(now_data) + '.pickle') == False):
        print("No problem file!")

    with open('./example-IS-e/pair' + str(now_data) + '.pickle', "rb") as f:
        problem = pickle.load(f)

    variable_features = problem[0]
    constraint_features = problem[1]
    edge_indices = problem[2]
    edge_feature = problem[3]

    patition_color = problem[4]
    optimal_solution = problem[5]
    #edge, features, labels, idx_train = load_data()

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
    num_label = [0, 0]

    for item in optimal_solution:
        num_label[1 - item] += 1
    num_label[0] /= n
    #num_label[0] -= 0.2
    num_label[1] /= n
    #num_label[1] += 0.2
    print(num_label)
    num_label = [1, 1]
    num_label = torch.as_tensor(num_label).to(device)
    data_labels.append(num_label)

    for i in range(m):
        optimal_solution.append(0)

    labels = torch.as_tensor(optimal_solution)
    '''
    labels = []
    for i in range(n + m):
        if(optimal_solution[i] == 0):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    labels = torch.tensor(labels)
    '''
    data_solution.append(labels)

    idx_train = torch.as_tensor(range(n))
    data_idx_train.append(idx_train)


#print(idx_train[3])

#edge = torch.tensor(edge_indices).t()
#labels, idx_train, idx_val, idx_test = load_data()

#feature：二维矩阵，含n个向量每个点一个，每个向量m维（features.shape[1]）


# Model and optimizer
model = SpGAT(nfeat=data_features[0].shape[1],    #特征维度
            nhid=args.hidden,             #隐藏层每一层的特征维度
            nclass=int(data_solution[0].max()) + 1, #类别数
            dropout=args.dropout,         #dropout
            nheads=args.nb_heads,         #头数
            alpha=args.alpha)             #LeakyReLU alpha系数

optimizer = optim.Adam(model.parameters(),    
                       lr=args.lr,                        #learning rate
                       weight_decay=args.weight_decay)    #weight decay(权值衰减) 防止过拟合


if args.cuda: #丢到GPU上
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
    #定义计算图，实现自动求导


def train(epoch, num):
    global data_edge_features
    #print(data_edge_features[num])
    t = time.time()
    #print(data_features[num])
    #print(data_edge[num])
    #print(data_edge_features[num])
    output, data_edge_features[num] = model(data_features[num], data_edge_A[num], data_edge_B[num], data_edge_features[num].detach())
    #lf = Focal_Loss(torch.as_tensor(data_labels[num]))
    #loss_train=lf(output[idx_train], data_solution[num][idx_train])
    #print(loss_train.data.item())

    
    
    ''' 
    idx_train = []
    for i in range(n):
        if(optimal_solution[i] == 0 and random.random() <= num_label[0]):
            idx_train.append(i)
        if(optimal_solution[i] == 1 and random.random() <= num_label[1]):
            idx_train.append(i)
    '''

    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    
    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    lf = Focal_Loss(torch.as_tensor(data_labels[num]))
    loss_train=lf(output[idx_train], data_solution[num][idx_train])
    #print(loss_train.data.item())
    #print(output)

    #loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    #loss_train = loss(output[idx_train], labels[idx_train].float())
    


   

    return loss_train

'''
def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
'''

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

    for name, parms in model.named_parameters():	
        print('\nBefore backward\n')
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("===========================")
    now_loss.backward()
    for name, parms in model.named_parameters():	
        print('\nAfter backward\n')
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("===========================")
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(now_loss))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:  #连续若干回合没有改善就停止了
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

print(loss_values)


'''
# Testing
compute_test()
'''


