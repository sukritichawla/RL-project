"""Compact Snake environment shared by all agents."""
import random
import numpy as np
DIRECTIONS=[(0,-1),(1,0),(0,1),(-1,0)]

class SnakeEnv:
    def __init__(self,width=12,height=12,seed=None,max_steps=500):
        self.width=width; self.height=height; self.rng=random.Random(seed); self.max_steps=max_steps
        self.reset()
    def reset(self):
        x,y=self.width//2,self.height//2
        self.snake=[(x,y),(x-1,y),(x-2,y)]; self.direction=1; self.score=0; self.steps=0; self.steps_since_food=0; self.done=False
        self.food=self._place_food(); return self.get_state()
    def _place_food(self):
        free=[(x,y) for x in range(self.width) for y in range(self.height) if (x,y) not in self.snake]
        return self.rng.choice(free) if free else None
    def _ahead(self,direction):
        dx,dy=DIRECTIONS[direction]; x,y=self.snake[0]; return x+dx,y+dy
    def _collision(self,pos,allow_tail=False):
        x,y=pos
        if x<0 or x>=self.width or y<0 or y>=self.height: return True
        body=self.snake[:-1] if allow_tail else self.snake
        return pos in body
    def _danger(self,direction):
        nxt=self._ahead(direction)
        return self._collision(nxt,allow_tail=(nxt!=self.food))
    def get_state(self):
        x,y=self.snake[0]; fx,fy=self.food if self.food is not None else (x,y)
        return np.asarray([self._danger(self.direction),self._danger((self.direction+1)%4),self._danger((self.direction-1)%4),*[self.direction==i for i in range(4)],fx<x,fx>x,fy<y,fy>y],dtype=np.float32)
    @staticmethod
    def state_key(state):
        key=0
        for bit in state: key=(key<<1)|int(bit)
        return key
    def step(self,action):
        if self.done: raise RuntimeError('Episode ended; call reset().')
        if action==1: self.direction=(self.direction+1)%4
        elif action==2: self.direction=(self.direction-1)%4
        nxt=self._ahead(self.direction); ate=nxt==self.food
        if self._collision(nxt,allow_tail=not ate):
            self.done=True; self.steps+=1
            return self.get_state(),-10.0,True,{'score':self.score,'steps':self.steps,'collision':True}
        self.snake.insert(0,nxt); self.steps+=1
        if ate:
            self.score+=1; self.steps_since_food=0; self.food=self._place_food(); reward=10.0
            if self.food is None: self.done=True
        else:
            self.snake.pop(); self.steps_since_food+=1; reward=-0.01
        if self.steps_since_food>self.max_steps or self.steps>=self.max_steps:
            self.done=True; reward=-2.0
        return self.get_state(),reward,self.done,{'score':self.score,'steps':self.steps,'collision':False,'ate':ate}
