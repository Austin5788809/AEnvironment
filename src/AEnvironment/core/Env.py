import torch
import pygame
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
        参数action是一个0维tensor，表示pred的argmax结果，类型是torch.long
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
                pygame.init()
                pygame.font.init()
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
        if len(pack) != 5:
            raise ValueError(f"Environment step must return 5 values [state, action, reward, next_state, done], got {len(pack)}")

        s, a, reward, next_state, done = pack
        pack = [
            torch.as_tensor(s, dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.long),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(next_state, dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.long),
        ]
        if state is None: self.state = pack[3] # 当参数state有值时我不希望修改原环境，此时相当于直接调用_transition
        return pack
