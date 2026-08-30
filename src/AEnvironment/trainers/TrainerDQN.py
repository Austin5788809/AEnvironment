from AEnvironment.core import _TrainerBase
from collections import deque
import torch
import copy
import random
from typing import Iterator
class TrainerDQN(_TrainerBase):
    '''
    DQN适用于离散动作空间的强化学习
    模型的任务是预测当前状态下每个动作的Q值
    目标Q值的计算公式是Q_target(s, a) = r + gamma * Q_target(s', best_a')，Q_target目标网络给出的Q值
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

    def __call__(self, eps:float, do_render=False) -> tuple[int, float, int]:
        self.env.render_mode(do_render)
        self.env.render()
        state = self.env.reset().to(self.device) # 重启，并获取当前状态
        steps = 0
        s_reward = 0
        while True:
            legal_actions = self.env.legal_actions().to(self.device) # 获取所有可用动作
            action = torch.tensor(0, dtype=torch.long) # 我先声明一下
            if random.random() < eps: # eps为探索率，决定是听从模型还是随机走
                action = torch.multinomial(legal_actions.float(), 1).to(self.device)
            else:
                pred = self.model(state.unsqueeze(0)).squeeze(0).to(self.device) # 获取模型预测的所有q值
                pred = pred.masked_fill(~legal_actions, -float('inf')) # 把不可用动作的q值变成-inf
                action = pred.argmax().to(self.device) # type:torch.Tensor
            pack = self.env.step(action) # 获取step返回的五元组
            self.D.append(pack)
            _, _, reward, state, done = pack # 解包，顺便更新state，此时state存的是下一个状态，而不是当前状态
            state.to(self.device)
            
            if len(self.D) >= self.batch_size: # 当经验池大小足够时，开始训练
                batch_list = random.sample(self.D, self.batch_size) # 随机取出batch_size条经验
                s = torch.stack([exp[0] for exp in batch_list]).to(self.device)
                a = torch.stack([exp[1].reshape(-1) for exp in batch_list]).to(self.device)
                r = torch.stack([exp[2] for exp in batch_list]).to(self.device)
                sn = torch.stack([exp[3] for exp in batch_list]).to(self.device)
                d = torch.stack([exp[4] for exp in batch_list]).to(self.device)

                pred = self.model(s) # type: torch.Tensor
                Q = pred.gather(1, a).squeeze(1)  # a 的形状是 [batch, 1]
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
                return steps, s_reward, int(done.item())
