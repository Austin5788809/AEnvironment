from AEnvironment.trainers import TrainerDQN
from AEnvironment.core import Env
import torch
from numpy import sign

class MyEnv(Env): # 每当state为随机整数，模型有四个动作，分别可以使state+10/+1/-1/-10，模型目标是使state接近0
    def __init__(self):
        super().__init__()
        self.state = torch.tensor([0.0])

    def reset(self):
        self.state = torch.randint(-50, 50, (1,), dtype=torch.float)
        return self.state

    def legal_actions(self):
        return torch.tensor([1, 1, 1, 1], dtype=torch.bool) 

    def _transition(self, action : torch.Tensor, state : torch.Tensor):
        staten = state.clone()
        if action == 0:
            staten += 10
        elif action == 1:
            staten += 1
        elif action == 2:
            staten -= 1
        elif action == 3:
            staten -= 10
        reward = torch.tensor(0, dtype=torch.float)
        if staten == 0:
            reward = 2
        elif abs(staten) < abs(state) and sign(staten) == sign(state):
            reward = 1
        else:
            reward = -1
        return [state, action, reward, staten, torch.tensor(staten == 0, dtype=torch.long)]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.f(x)

model = Model()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
env = MyEnv()
trainer = TrainerDQN(model, optim, env, batch_size=32)
eps = 1.0

for i in range(1000):
    step, reward, done = trainer(eps)
    eps = max(0.01, eps - 0.001)
    print(f"step {step}, reward {reward}, done {done}, eps {eps}")

torch.save(model.state_dict(), "model.pth")

