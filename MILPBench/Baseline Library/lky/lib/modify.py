import torch
from typing import Type, cast, Self
from .mod import Component, Predict2Modify, Modify2Search, MScores, PScores, Cansol
        

class Modify(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "np":
            cls = Np
        elif component == "default":
            cls = Default
        else:
            raise ValueError("Modify component type is not defined")
        return super().__new__( cast(type[Self], cls) )
        
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)
            
    def work(self, input: Predict2Modify) -> Modify2Search:...
        
class Np(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        
        if taskname == "IP": 
            dhp = (400, 5)
        elif taskname == "IS":
            dhp = (300, 300) 
        elif taskname == "WA":
            dhp = (0, 600) 
        elif taskname == "CA":
            dhp = (400, 0)
        else :
            dhp = (0, 0)
        
        self.k0 = kwargs.get("k0", dhp[0])
        self.k1 = kwargs.get("k1", dhp[1])
        # tackle parameters    

    def work(self, input: PScores) -> MScores:
        self.begin()
        b_vars = input.b_vars
        scores = input.scores
        binary_name = [scores[i][1] for i in b_vars]
        # get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(scores)):
            type = "C"
            if scores[i][1] in binary_name:
                type = 'BINARY'
            scores[i] += [-1, type]

        scores=[x for x in scores if x[4]=='BINARY'] # get binary

        scores.sort(key=lambda x:x[2], reverse=True)

        # fixing variable picked by confidence scores
        count1 = 0
        for i in range(len(scores)):
            if count1 < self.k1:
                scores[i][3] = 1
                count1 += 1

        scores.sort(key=lambda x: x[2], reverse=False)
        count0 = 0
        for i in range(len(scores)):
            if count0 < self.k0:
                scores[i][3] = 0
                count0 += 1
        self.end()
        return MScores(scores)
        
class Default(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        # tackle parameters    

    def work(self, input: Cansol) -> Cansol:
        self.begin()
        self.end()
        return input
        