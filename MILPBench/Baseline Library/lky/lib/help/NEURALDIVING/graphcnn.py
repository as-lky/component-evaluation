from ftplib import B_CRLF
import numpy as np
import torch
import torch_geometric

__all__ = ["GNNPolicy"]

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output

# 注意 三分图的卷积结构可以使用二分图的卷积结构来构成

class GNNPolicy(torch.nn.Module):
    def __init__(self, random_feature=False, tripartite=False):
        super().__init__()
        self.tripartite = tripartite
        emb_size = 64
        cons_nfeats = 3 if random_feature else 2
        edge_nfeats = 1
        var_nfeats = 7 if random_feature else 6
        
        if tripartite:
            obj_nfeats = 5 if random_feature else 4

            # OBJ EMBEDDING
            self.obj_embedding = torch.nn.Sequential(
                torch.nn.LayerNorm(obj_nfeats),
                torch.nn.Linear(obj_nfeats, emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.ReLU(),
            )

            # EDGE1 EMBEDDING
            self.edge_embedding1 = torch.nn.Sequential(
                torch.nn.LayerNorm(edge_nfeats),
            )
            
            # EDGE2 EMBEDDING
            self.edge_embedding2 = torch.nn.Sequential(
                torch.nn.LayerNorm(edge_nfeats),
            )


        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        if tripartite:
            self.conv_c_to_o = BipartiteGraphConvolution()
            self.conv_o_to_c = BipartiteGraphConvolution()
            self.conv_v_to_o = BipartiteGraphConvolution()
            self.conv_o_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
            #torch.nn.Sigmoid()
        )

        self.output_select = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
            #torch.nn.Sigmoid()
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features, obj_features=None, obj_variable_val=None, obj_constraint_val=None, edge_obj_var=None, edge_obj_cons=None
    ):
        if self.tripartite:
            reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

            # First step: linear embedding layers to a common dimension (64)
            constraint_features = self.cons_embedding(constraint_features)
            edge_features = self.edge_embedding(edge_features)
            variable_features = self.var_embedding(variable_features)
            obj_features = self.obj_embedding(obj_features)
            obj_variable_val = self.edge_embedding1(obj_variable_val)
            obj_constraint_val = self.edge_embedding2(obj_constraint_val)


            edge_var_obj = torch.stack([edge_obj_var[1], edge_obj_var[0]], dim=0)
            edge_cons_obj = torch.stack([edge_obj_cons[1], edge_obj_cons[0]], dim=0)

            # Two half convolutions
            for i in range(3):
                obj_features = self.conv_v_to_o(variable_features, edge_var_obj, obj_variable_val, obj_features)

                
                constraint_features = self.conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                    )
                
                constraint_features = self.conv_o_to_c(
                    obj_features, edge_obj_cons, obj_constraint_val, constraint_features
                )
                
                obj_features = self.conv_c_to_o(
                    constraint_features, edge_cons_obj, obj_constraint_val, obj_features
                )
                
                variable_features = self.conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )
                
                variable_features = self.conv_o_to_v(
                    obj_features, edge_obj_var, obj_variable_val, variable_features
                )
            # A final MLP on the variable features
            # print(variable_features.shape)
            output = self.output_module(variable_features).squeeze(-1)
            select = self.output_select(variable_features).squeeze(-1)
            return output, select
        else:
            reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

            # First step: linear embedding layers to a common dimension (64)
            constraint_features = self.cons_embedding(constraint_features)
            edge_features = self.edge_embedding(edge_features)
            variable_features = self.var_embedding(variable_features)

            # Two half convolutions
            for i in range(3):
                
                constraint_features = self.conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                    )
                
                variable_features = self.conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )
                
            # A final MLP on the variable features
            # print(variable_features.shape)
            output = self.output_module(variable_features).squeeze(-1)
            select = self.output_select(variable_features).squeeze(-1)
            return output, select
