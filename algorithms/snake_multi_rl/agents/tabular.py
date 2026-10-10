"""From-scratch tabular Q-Learning and SARSA."""
import numpy as np
class TabularAgent:
    def __init__(self,algorithm,n_actions=3,alpha=.12,gamma=.95,epsilon=1.,epsilon_min=.03,epsilon_decay=.995,seed=42):
        assert algorithm in ('q_learning','sarsa')
        self.algorithm=algorithm; self.n_actions=n_actions; self.alpha=alpha; self.gamma=gamma
        self.epsilon=epsilon; self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay; self.rng=np.random.default_rng(seed); self.q={}
    def values(self,key):
        if key not in self.q: self.q[key]=np.zeros(self.n_actions,dtype=np.float32)
        return self.q[key]
    def act(self,key,explore=True):
        if explore and self.rng.random()<self.epsilon: return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.values(key)))
    def update(self,s,a,r,ns,done,next_action=None):
        q=self.values(s)
        if done: target=r
        elif self.algorithm=='q_learning': target=r+self.gamma*float(np.max(self.values(ns)))
        else:
            if next_action is None: raise ValueError('SARSA requires next_action on nonterminal steps')
            target=r+self.gamma*float(self.values(ns)[next_action])
        q[a]+=self.alpha*(target-q[a])
    def decay_epsilon(self): self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)
    def save(self,path):
        keys=np.array(list(self.q),dtype=np.int64); vals=np.array([self.q[k] for k in keys],dtype=np.float32)
        np.savez_compressed(path,keys=keys,values=vals,epsilon=[self.epsilon])
    def load(self,path):
        with np.load(path) as d: self.q={int(k):v.copy() for k,v in zip(d['keys'],d['values'])}; self.epsilon=float(d['epsilon'][0])
