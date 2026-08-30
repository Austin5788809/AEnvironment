from core import _TrainerBase
from collections import deque
import torch
import copy
import random
from typing import Iterator#aba
class TrainerDoubleDQN(_TrainerBase):
    '''
    与DQN相似，但Q_target(s', best_a')的best_a'用主网络选取而不是目标网络
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
                action = torch.multinomial(legal_actions.float(), 1).to(self.device)
            else:
                pred = self.model(state.unsqueeze(0)).squeeze(0).to(self.device) # 获取模型预测的所有q值
                pred = pred.masked_fill(~legal_actions, -float('inf')) # 把不可用动作的q值变成-inf
                action = pred.argmax() # type:torch.Tensor
            pack = self.env.step(action) # 获取step返回的五元组
            self.D.append(pack)
            _, _, reward, state, done = pack # 解包，顺便更新state，此时state存的是下一个状态，而不是当前状态
            state.to(self.device)

            if len(self.D) >= self.batch_size: # 当经验池大小足够时，开始训练
                batch_list = random.sample(self.D, self.batch_size) # 随机取出batch_size条经验
                s = torch.stack([exp[0] for exp in batch_list]).to(self.device)
                a = torch.stack([exp[1] for exp in batch_list]).to(self.device)
                r = torch.stack([exp[2] for exp in batch_list]).to(self.device)
                sn = torch.stack([exp[3] for exp in batch_list]).to(self.device)
                d = torch.stack([exp[4] for exp in batch_list]).to(self.device)

                pred = self.model(s) # type: torch.Tensor
                Q = pred.gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = self.model(sn).argmax(dim=1)  # 当前网络选动作
                    target_pred = self.target(sn)
                    max_next_Q = target_pred.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    Q_tgt = r + self.gamma * (1.0 - (d != 0).float()) * max_next_Q
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