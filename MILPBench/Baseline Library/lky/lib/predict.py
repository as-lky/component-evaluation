import torch
from typing import Type
from mod import Component, Graphencode2Predict, Predict2Modify

class Predict(Component):
    def __new__(cls, component, *args, **kwargs):
        if component == "gcn":
            return super().__new__(GCN, *args, **kwargs)
        else :
            raise ValueError("Predict component type is not defined")

class GCN(Predict):
    def __init__(self, component, *args, **kwargs):
        ... # tackle parameters
        
    def work(self, input: Type[Graphencode2Predict]) -> Type[Predict2Modify]:        
        