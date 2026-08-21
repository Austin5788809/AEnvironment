from utils import *
import torch
from collections import deque
import copy
import random

class Env:
    '''
    环境基类
    用户需要自己继承这个基类，然后补全具体逻辑，或者使用预设
    请调用super().__init__()来生成成员
    '''
    def __init__(self):
        self.sreen = None

    def do_render(self, do):
        '''传入do代表是否渲染'''

    def render(self):
        pass

    def reset(self) -> torch.Tensor:
        '''
        使环境重置到初始状态
        返回state，类型为tensor，dtype=torch.float32
        '''

    def legal_action(self) -> torch.Tensor:
        '''
        返回所有可用动作的掩码，类型为torch.bool
        '''

    def step(self, action:torch.Tensor) -> list[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        '''
        返回一个list，包含五个tensor，分别代表state、action、reward、state'、done
        这些tensor的类型分别应是torch.float32、torch.long、torch.float32、torch.float32、torch.long
        其中action、reward、done是单元素tensor
        done为0时代表未结束，-1代表负，1代表胜
        '''

class __Trainer_base:
    '''
    训练器
    构造时传入需要训练的模型，自动训练
    '''
    def __init__(self, module:torch.nn.Module, optim:torch.optim.Optimizer, env:Env, device=torch.device("cpu"), batch_size=128):
        self.module = module
        self.optim = optim
        self.env = env
        self.device = torch.device
        self.batch_size=batch_size

    def __call__(self, eps:float, do_render=False):
        '''
        这是一个生成器
        每次调用时，进行一局模拟，然后返回日志
        日志包括步数、总reward、胜负
        '''

class Trainer_DQN(__Trainer_base):
    def __init__(self, module, optim, env, gamma, D_maxlen=10000):
        super().__init__(module, optim, env, batch_size)
        self.gamma = gamma
        self.D = deque(maxlen=D_maxlen)
        self.target = copy.deepcopy(self.module)

    def __call__(self, eps, do_render=False):
        state = self.env.reset() # 重启，并获取当前状态
        while True:
            legal_action = self.env.legal_action() # 获取所有可用动作
            action = 0
            if random.random() < eps: # eps为探索率，决定是听从模型还是随机走
                action = torch.tensor(random.choice([i for i, x in enumerate(list(legal_action)) if x]), dtype=torch.long)
                '''
                这句话有点复杂，拆解一下
                首先我生成了所有合法动作的下标[i for i, x in enumerate(list(legal_action)) if x]
                然后随机选择了一个random.choice(...)
                最后转换成了tensor类型torch.tensor(..., dtype=torch.long)
                '''
            else:
                pred = self.module(state) # 获取模型预测的q值
                pred = pred.masked_fill(~legal_action, -float('inf')) # 把不可用动作的q值变成-inf
                action = pred.argmax()
            pack = self.env.step(action) # 获取step返回的五元组
            self.D.append(pack)
            _, _, _, sn, d = pack
            state = sn # 更新状态
            # TODO:如果len(D)大于batch_size就开始训练
            if d.item() != 0:
                break
            
            