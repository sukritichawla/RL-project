"""Train Q-Learning, SARSA, DQN and Double DQN from scratch."""
import os,csv,random
import numpy as np
from config import EPISODES,MAX_STEPS_PER_EPISODE,SEED,RESULTS_DIR,MODELS_DIR
from snake_env import SnakeEnv
from agents.tabular import TabularAgent
from agents.deep import DQNAgent

def run_episode(env,agent,name,training=True):
    state=env.reset(); key=env.state_key(state); total=0.; steps=0
    action=agent.act(key,explore=training) if name=='sarsa' else None
    while steps<MAX_STEPS_PER_EPISODE:
        if name in ('q_learning','sarsa'):
            if name=='q_learning': action=agent.act(key,explore=training)
            ns,reward,done,info=env.step(action); nk=env.state_key(ns)
            na=agent.act(nk,explore=training) if name=='sarsa' and not done else None
            if training: agent.update(key,action,reward,nk,done,na)
            state,key=ns,nk
            if name=='sarsa' and not done: action=na
        else:
            action=agent.act(state,explore=training); ns,reward,done,info=env.step(action)
            if training: agent.remember(state,action,reward,ns,done); agent.learn()
            state=ns
        total+=reward; steps+=1
        if done: break
    if training: agent.decay_epsilon()
    return {'reward':total,'score':env.score,'steps':steps,'epsilon':agent.epsilon}

def main():
    os.makedirs(RESULTS_DIR,exist_ok=True); os.makedirs(MODELS_DIR,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED)
    agents={'q_learning':TabularAgent('q_learning',seed=SEED),'sarsa':TabularAgent('sarsa',seed=SEED+1),'dqn':DQNAgent('dqn',seed=SEED+2),'double_dqn':DQNAgent('double_dqn',seed=SEED+3)}
    rows=[]
    for name,agent in agents.items():
        print(f'\n=== {name}: {EPISODES} episodes ===',flush=True); env=SnakeEnv(seed=SEED,max_steps=MAX_STEPS_PER_EPISODE)
        for ep in range(1,EPISODES+1):
            stats=run_episode(env,agent,name,True); rows.append({'algorithm':name,'episode':ep,**stats})
            if ep==1 or ep%50==0:
                recent=[r['score'] for r in rows if r['algorithm']==name and r['episode']>ep-50]
                print(f'{name:12s} {ep:4d}/{EPISODES} | mean score last 50={np.mean(recent):.2f} | reward={stats["reward"]:.2f} | epsilon={agent.epsilon:.3f}',flush=True)
        path=os.path.join(MODELS_DIR,name+('.npz' if name in ('q_learning','sarsa') else '.pt')); agent.save(path); print('Saved',path)
    with open(os.path.join(RESULTS_DIR,'training_metrics.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['algorithm','episode','reward','score','steps','epsilon']); w.writeheader(); w.writerows(rows)
    print('Saved results/training_metrics.csv. Next run: python evaluate.py')
if __name__=='__main__': main()
