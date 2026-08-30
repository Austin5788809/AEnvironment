from typing import Any

from AEnvironment.core import Env
from AEnvironment.trainers import TrainerDQN
import torch
import random

class SimpleEnv(Env):
    """一个极简任务：给定 [a, b]，选择动作 0/1，选中正确的数字才给 reward。"""
    def __init__(self, do_render=False, screen_size=(800, 600)):
        super().__init__(do_render, screen_size)

    def reset(self) -> torch.Tensor:
        # 生成一个简单状态，包含两个数字和它们的和
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        self.state = torch.tensor([float(a), float(b), float(a + b)], dtype=torch.float32)
        return self.state

    def legal_actions(self) -> torch.Tensor:
        # 两个动作都合法
        return torch.tensor([True, True], dtype=torch.bool)

    def _transition(self, action: torch.Tensor, state: torch.Tensor) -> list[torch.Tensor]:
        # 这里动作 0 表示选择左边，动作 1 表示选择右边
        # 但真正的答案直接来自 a+b，任务是“看状态，输出正确动作”
        # 我们给出奖励：动作 0/1 都是从状态中做一个二值选择，正确则 +1，否则 -1
        target = 0 if state[0] + state[1] < 10 else 1
        correct = 1 if int(action.item()) == target else 0
        reward = 1.0 if correct else -1.0
        done = 1
        return [state, action, torch.tensor(reward, dtype=torch.float32), state.clone(), torch.tensor(done, dtype=torch.long)]

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

env = SimpleEnv(do_render=False)
model = Model()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = TrainerDQN(model, optim, env)

# 训练测试：如果训练有效，reward 应该逐渐变正
for _ in range(100):
    step, reward, done = trainer(0.9)
    print(f"step={step}, reward={reward}, done={done}")

# 保存模型
torch.save(model.state_dict(), "test_model_simple.pth")
print("训练完成")
