import os
from .mod import Component, Preprocess2Graphencode
from typing import Type

class Preprocess(Component): # no usage for now
    def __init__(self, device, taskname, instance, sequence_name):
        super().__init__(device, taskname, instance, sequence_name)
    
    def work(self) -> Preprocess2Graphencode:
        
        print('Preprocess Component is working ...')
        # sn = str(self.sequence_name)
        # if not os.path.isdir(f'./logs'):
        #     os.mkdir(f'./logs')
        # if not os.path.isdir(f'./logs/{self.taskname}'):
        #     os.mkdir(f'./logs/{self.taskname}')
        # if not os.path.isdir(f'./logs/{self.taskname}/{sn}'):
        #     os.mkdir(f'./logs/{self.taskname}/{sn}')
        # if not os.path.isdir(f'./logs/{self.taskname}/{sn}/work'):
        #     os.mkdir(f'./logs/{self.taskname}/{sn}/work')

        print('Preprocess Component is done.')
        
        return Preprocess2Graphencode()