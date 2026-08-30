from utils import *
from collections import deque
import copy
import random
from typing import Iterator
from abc import ABC, abstractmethod

_zero = torch.tensor(0) # 占位符，用于函数返回

class Env(ABC):
    '''
    环境基类
    用户需要自己继承这个基类，然后补全具体逻辑，或者使用预设
    请调用super().__init__()来生成成员

    ```python
    class Your_env(Env):
        def __init__(self):
            super().__init__()
            # 其它初始化
        
        def render(self):
            # 渲染逻辑

        def reset(self) -> torch.Tensor:
            # 重置逻辑
            return state
        
        def legal_actions(self) -> torch.Tensor:
            # 返回所有可用动作的掩码
            return legal_action_mask
        
        def step(self, action:torch.Tensor) -> list[torch.Tensor]:
            # 执行动作逻辑
            return [state, action, reward, staten, done]
        
    your_env = Your_env()
    ```
    '''
    def __init__(self, do_render=False, screen_size=(800, 600)):
        self.do_render = do_render # 是否渲染
        self.screen = None # 当启用渲染时，类型为pygame.Surface
        self.screen_size = screen_size
        self.render_mode(do_render)
        self.state = None # 用户的所有状态都要转换成一个tensor并装在这个变量中

    # ----------用户需要重写的方法------------
    # @abstractmethod，如果不需要渲染的话不需要重写这个
    def _render(self): 
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        '''
        使环境重置到初始状态
        返回state，类型为tensor，dtype=torch.float32
        ```python
        state = your_env.reset()
        while True:
            # 其它逻辑
        ```
        '''
        return _zero

    @abstractmethod
    def legal_actions(self) -> torch.Tensor:
        '''
        返回所有可用动作的掩码，类型为torch.bool
        ```python
        legal_actions = your_env.legal_actions()
        ```
        '''
        return _zero

    @abstractmethod
    def _transition(self, action: torch.Tensor, state: torch.Tensor) -> list[torch.Tensor]:
        '''
        用户需要实现这个函数，而不是step，这个函数需要根据当前状态和动作返回[s, a, r, s', done]五个tensor
        这些tensor的类型分别应是torch.float32、torch.long、torch.float32、torch.float32、torch.long
        其中action、reward、done是单元素tensor
        done为0时代表未结束，-1代表负，1代表胜
        有4个step的重载将调用这个函数
        这个函数不需要改变类的状态
        '''
        return [_zero, _zero, _zero, _zero, _zero]

    # ----------用户无需重写的方法------------
    def render_mode(self, do_render:bool):
            '''
            设置是否渲染
            ```python
            your_env.render_mode(True)
            ```
            '''
            self.do_render = do_render
            if do_render:
                if self.screen is None:
                    pygame.display.init()
                    self.screen = pygame.display.set_mode(self.screen_size)
                else:
                    pygame.display.set_mode(self.screen_size)
            else:
                if self.screen is not None:
                    pygame.display.quit()
                    self.screen = None

    def render(self):
        if self.do_render:
            self._render()
    
    def step(self, action, state:torch.Tensor|None=None) -> list[torch.Tensor]:
        '''
        返回一个list，[s, a, r, s', done]
        '''
        action = torch.as_tensor(action, dtype=torch.long)
        if state is None: state = self.state

        assert isinstance(state, torch.Tensor)
        pack = self._transition(action, state)
        if state is None: self.state = pack[3] # 当参数state有值时我不希望修改原环境，此时相当于直接调用_transition
        return pack



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
    def __call__(self, eps:float, do_render=False) -> Iterator[tuple[int, float, int]]:
        '''
        这需要是一个生成器
        每次调用时，进行一局模拟，然后返回日志
        日志包括步数、总reward、胜负
        需要用用户重写
        '''
        yield 0, 0.0, 0

class TrainerDQN(_TrainerBase):
    '''
    DQN适用于离散动作空间的强化学习
    模型的任务是预测当前状态下每个动作的Q值
    目标Q值的计算公式是Q_target(s, a) = r + gamma * Q_target(s', a')，Q_target目标网络给出的Q值
    大意为，从现在到未来的全部reward总和，但是越未来的reward，影响力越小，因为有折扣因子gamma
    我们需要让网络给出的Q值尽量与目标Q值保持一致，也就是说，loss = MSE(Q(s, a), Q_target(s, a))
    ```python
    trainer = TrainerDQN(your_model, your_optim, your_env)
    eps = 1.0
    for pack in trainer(eps, True):
        # ...
        eps = max(0.1, eps*0.9999)
    ```
    '''
    def __init__(self, model, optim, env, gamma=0.99, update_step=300, D_maxlen=10000, batch_size=128, device=torch.device("cpu")):
        super().__init__(model, optim, env, batch_size, device)
        self.gamma = gamma
        self.D = deque(maxlen=D_maxlen)
        self.target = copy.deepcopy(self.model)
        self.total_steps = 0
        self.update_step = update_step
        self.MSEloss = torch.nn.MSELoss()

    def __call__(self, eps:float, do_render=False) -> Iterator[tuple[int, float, int]]:
        self.env.render_mode(do_render)
        self.env.render()
        state = self.env.reset().to(self.device) # 重启，并获取当前状态
        steps = 0
        s_reward = 0
        while True:
            legal_actions = self.env.legal_actions().to(self.device) # 获取所有可用动作
            action = torch.tensor(0, dtype=torch.long) # 我先声明一下
            if random.random() < eps: # eps为探索率，决定是听从模型还是随机走
                action = torch.multinomial(legal_actions.float(), 1)
            else:
                pred = self.model(state.unsqueeze(0)).squeeze(0) # 获取模型预测的所有q值
                pred = pred.masked_fill(~legal_actions, -float('inf')) # 把不可用动作的q值变成-inf
                action = pred.argmax() # type:torch.Tensor
            pack = self.env.step(action) # 获取step返回的五元组
            self.D.append(pack)
            _, _, reward, state, done = pack # 解包，顺便更新state，此时state存的是下一个状态，而不是当前状态
            
            if len(self.D) >= self.batch_size: # 当经验池大小足够时，开始训练
                batch_list = random.sample(self.D, self.batch_size) # 随机取出batch_size条经验
                s = torch.stack([exp[0] for exp in batch_list]).to(self.device)
                a = torch.stack([exp[1] for exp in batch_list]).to(self.device)
                r = torch.stack([exp[2] for exp in batch_list]).to(self.device)
                sn = torch.stack([exp[3] for exp in batch_list]).to(self.device)
                d = torch.stack([exp[4] for exp in batch_list]).to(self.device)

                pred = self.model(s) # type: torch.Tensor
                Q = pred.gather(1, a.unsqueeze(1)).squeeze(1) # 最烦来回对齐维度啊啊啊
                with torch.no_grad():
                    pred_tgt_n = self.target(sn) # type: torch.Tensor
                    Q_tgt_n, _ = pred_tgt_n.max(1, keepdim=False)
                    Q_tgt = r + self.gamma * Q_tgt_n * (1.0 - (d != 0).float())
                loss = self.MSEloss(Q, Q_tgt)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
            if self.total_steps % self.update_step == 0:
                self.target.load_state_dict(self.model.state_dict())
            steps += 1
            self.total_steps += 1
            s_reward += reward.item()
            self.env.render()
            if done.item() != 0:
                yield steps, s_reward, int(done.item())
                break
            
            