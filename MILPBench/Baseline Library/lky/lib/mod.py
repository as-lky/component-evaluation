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
    def __init__(self, device, taskname, instance_folder):
        self.device = device
        self.taskname = taskname
        self.instance_folder = instance_folder

class Preprocess2Graphencode(LayerConvey):
    def __init__(self, device, taskname, instance_folder):
        self.device = device
        self.taskname = taskname
        self.instance_folder = instance_folder
        
class Graphencode2Predict(LayerConvey):
    def __init__(self, device, taskname, instance_folder, A, v_map, v_nodes, c_nodes, b_vars):
        self.device = device
        self.taskname = taskname
        self.instance_folder = instance_folder
        self.A = A
        self.v_map = v_map
        self.v_nodes = v_nodes
        self.c_nodes = c_nodes
        self.b_vars = b_vars
        
class Predict2Modify(LayerConvey):
    def __init__(self, device, taskname, instance_folder):
        self.device = device
        self.taskname = taskname
        self.instance_folder = instance_folder
        
class Modify2Search(LayerConvey):
    def __init__(self, device, taskname, instance_folder):
        self.device = device
        self.taskname = taskname
        self.instance_folder = instance_folder
        

        

