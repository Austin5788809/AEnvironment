import torch
from typing import Iterator
from abc import ABC, abstractmethod
from AEnvironment.core.Env import Env

class _TrainerBase(ABC):
    '''
    训练器基类
    构造时传入需要训练的模型，自动训练
    一般不允许用户自定义训练器
    '''
    def __init__(self, module:torch.nn.Module, optim:torch.optim.Optimizer, env:Env, batch_size=128, device=torch.device("cpu")):
        self.model = module
        self.optim = optim
        self.env = env
        self.device = device
        self.batch_size=batch_size

    @abstractmethod
    def __call__(self, eps:float, do_render=False) -> tuple[int, float, int]:
        '''
        每次调用时，进行一局模拟，然后返回日志
        日志包括步数、总reward、胜负
        需要用用户重写
        '''
        return 0, 0.0, 0
