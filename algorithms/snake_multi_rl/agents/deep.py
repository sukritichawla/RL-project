"""From-scratch DQN and Double DQN update logic (PyTorch supplies neural layers)."""
import random
from collections import deque
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
class QNetwork(nn.Module):
    def __init__(self,state_dim=11,n_actions=3):
        super().__init__(); self.net=nn.Sequential(nn.Linear(state_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,n_actions))
    def forward(self,x): return self.net(x)
class ReplayBuffer:
    def __init__(self,capacity=50000): self.memory=deque(maxlen=capacity)
    def add(self,*transition): self.memory.append(tuple(transition))
    def sample(self,n):
        b=random.sample(self.memory,n); s,a,r,ns,d=zip(*b)
        return np.asarray(s,np.float32),np.asarray(a,np.int64),np.asarray(r,np.float32),np.asarray(ns,np.float32),np.asarray(d,np.float32)
    def __len__(self): return len(self.memory)
class DQNAgent:
    def __init__(self,algorithm='dqn',state_dim=11,n_actions=3,lr=1e-3,gamma=.95,epsilon=1.,epsilon_min=.03,epsilon_decay=.995,batch_size=64,buffer_size=50000,target_update=250,seed=42,device=None):
        assert algorithm in ('dqn','double_dqn'); self.algorithm=algorithm; self.n_actions=n_actions; self.gamma=gamma
        self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay; self.batch_size=batch_size; self.target_update=target_update; self.steps=0
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.online=QNetwork(state_dim,n_actions).to(self.device); self.target=QNetwork(state_dim,n_actions).to(self.device); self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr); self.replay=ReplayBuffer(buffer_size)
    def act(self,state,explore=True):
        if explore and np.random.random()<self.epsilon: return int(np.random.randint(self.n_actions))
        with torch.no_grad(): return int(self.online(torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)).argmax(1).item())
    def remember(self,s,a,r,ns,d): self.replay.add(np.array(s,copy=True),a,r,np.array(ns,copy=True),d)
    def learn(self):
        if len(self.replay)<self.batch_size: return None
        s,a,r,ns,d=self.replay.sample(self.batch_size)
        s=torch.as_tensor(s,device=self.device); a=torch.as_tensor(a,device=self.device).unsqueeze(1); r=torch.as_tensor(r,device=self.device); ns=torch.as_tensor(ns,device=self.device); d=torch.as_tensor(d,device=self.device)
        current=self.online(s).gather(1,a).squeeze(1)
        with torch.no_grad():
            if self.algorithm=='double_dqn':
                selected=self.online(ns).argmax(1,keepdim=True); nxt=self.target(ns).gather(1,selected).squeeze(1)
            else: nxt=self.target(ns).max(1).values
            target=r+self.gamma*(1-d)*nxt
        loss=F.smooth_l1_loss(current,target); self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),10.); self.optimizer.step()
        self.steps+=1
        if self.steps%self.target_update==0: self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
    def decay_epsilon(self): self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)
    def save(self,path): torch.save({'algorithm':self.algorithm,'online':self.online.state_dict(),'target':self.target.state_dict(),'optimizer':self.optimizer.state_dict(),'epsilon':self.epsilon,'steps':self.steps},path)
    def load(self,path):
        ck=torch.load(path,map_location=self.device,weights_only=False); self.online.load_state_dict(ck['online']); self.target.load_state_dict(ck['target']); self.epsilon=float(ck.get('epsilon',self.epsilon)); self.steps=int(ck.get('steps',0))
