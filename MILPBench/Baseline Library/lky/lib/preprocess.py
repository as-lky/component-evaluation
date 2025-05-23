import os
import re
from .mod import Component, Preprocess2Graphencode
from typing import Type

class Preprocess(Component): # no usage for now
    def __init__(self, device, taskname, instance, sequence_name):
        super().__init__(device, taskname, instance, sequence_name)
    
    def work(self) -> Preprocess2Graphencode:
        
        print('Preprocess Component is working ...')
        
        instance_name = os.path.basename(self.instance)
        tmp = re.match(r"(.*)_[0-9]+\.lp", instance_name)
        tmp = tmp.group(1)
  #      sn = str(self.sequence_name)
        sn = ""
        for _ in self.sequence_name:
            sn += _ + "_"
        if not os.path.isdir(f'./logs'):
            os.mkdir(f'./logs')
        if not os.path.isdir(f'./logs/work'):
            os.mkdir(f'./logs/work')
        if not os.path.isdir(f'./logs/work/{self.taskname}'):
            os.mkdir(f'./logs/work/{self.taskname}')
        if not os.path.isdir(f'./logs/work/{self.taskname}/{sn}'):
            os.mkdir(f'./logs/work/{self.taskname}/{sn}')
        if not os.path.isdir(f'./logs/work/{self.taskname}/{sn}/{tmp}'):
            os.mkdir(f'./logs/work/{self.taskname}/{sn}/{tmp}')
        
        print('Preprocess Component is done.')
        
        return Preprocess2Graphencode()