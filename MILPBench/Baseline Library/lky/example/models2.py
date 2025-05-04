import torch
import torch.nn as nn
import torch.nn.functional as F
from layers2 import SpGraphAttentionLayer



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        embed_size = 64
        self.input_module = torch.nn.Sequential(
            torch.nn.Linear(nfeat, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
        )
        self.attentions_u_to_v = [SpGraphAttentionLayer(embed_size,
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_u_to_v):
            self.add_module('attention_u_to_v_{}'.format(i), attention)
        self.attentions_v_to_u = [SpGraphAttentionLayer(embed_size,
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_v_to_u):
            self.add_module('attention_v_to_u_{}'.format(i), attention)

        self.out_att_u_to_v = SpGraphAttentionLayer(nhid * nheads, 
                                               embed_size, 
                                               dropout=dropout, 
                                               alpha=alpha, 
                                               concat=False)
        self.out_att_v_to_u = SpGraphAttentionLayer(nhid * nheads, 
                                               embed_size, 
                                               dropout=dropout, 
                                               alpha=alpha, 
                                               concat=False)
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, nclass, bias=False),
            #torch.nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, edgeA, edgeB, edge_feat):
        #print(x)
        x = self.input_module(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        #print(x)
        new_edge = torch.cat([att(x, edgeA, edge_feat)[1] for att in self.attentions_u_to_v], dim=1)
        x = torch.cat([att(x, edgeA, edge_feat)[0] for att in self.attentions_u_to_v], dim=1)
        x = self.out_att_u_to_v(x, edgeA, edge_feat)
        new_edge = torch.mean(new_edge, dim = 1).reshape(new_edge.size()[0], 1)
        #x = self.softmax(x)
        new_edge_ = torch.cat([att(x, edgeB, new_edge)[1] for att in self.attentions_v_to_u], dim=1)
        x = torch.cat([att(x, edgeB, new_edge)[0] for att in self.attentions_v_to_u], dim=1)
        x = self.out_att_v_to_u(x, edgeB, new_edge)
        new_edge_ = torch.mean(new_edge_, dim = 1).reshape(new_edge_.size()[0], 1)
        #x = self.softmax(x)
        #x = self.out_att_3(x, edge, edge_feat)
        #print(x)
        #x = self.softmax(x)

        #x = self.out_att(x, edge, edge_feat)
        #x = self.softmax(x)
        x = self.output_module(x)
        x = self.softmax(x)

        return x, new_edge_

