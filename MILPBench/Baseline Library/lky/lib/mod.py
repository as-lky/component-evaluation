class Component:
    def __init__(self):
        pass

    def work(
        self, input, *args
    ):  # args are the component select parameters, which should be interpreted and correspended to the input
        output = 1
        return output

class LayerConvey:
    ...
# For Component, the parameters from outside are the select args; from front layer are the instance processing result


class Init2Preprocess(LayerConvey):
    def __init__(self, device, taskname, instance):
        self.device = device
        self.taskname = taskname
        self.instance = instance

class Preprocess2Graphencode(LayerConvey):
    def __init__(self, device, taskname, instance):
        self.device = device
        self.taskname = taskname
        self.instance = instance
    
class Graphencode2Predict(LayerConvey):
    def __init__(self, device, taskname, instance, constraint_features, edge_indices, edge_features, variable_features, v_map, v_nodes, c_nodes, b_vars):
        self.device = device
        self.taskname = taskname
        self.instance = instance
        self.constraint_features = constraint_features 
        self.edge_indices = edge_indices
        self.edge_features = edge_features
        self.variable_features = variable_features
        self.v_map = v_map
        self.v_nodes = v_nodes
        self.c_nodes = c_nodes
        self.b_vars = b_vars
        
class Predict2Modify(LayerConvey):
    def __init__(self, device, taskname, instance):
        self.device = device
        self.taskname = taskname
        self.instance = instance

        
class Modify2Search(LayerConvey):
    def __init__(self, device, taskname, instance):
        self.device = device
        self.taskname = taskname
        self.instance = instance

        

        

