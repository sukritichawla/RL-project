"""Evaluate trained agents on fresh episodes and generate comparison plots."""
import os,csv
import numpy as np
import matplotlib.pyplot as plt
from config import SEED,RESULTS_DIR,MODELS_DIR,MAX_STEPS_PER_EPISODE
from snake_env import SnakeEnv
from agents.tabular import TabularAgent
from agents.deep import DQNAgent
from train import run_episode

def load_agents():
    agents={'q_learning':TabularAgent('q_learning',seed=SEED),'sarsa':TabularAgent('sarsa',seed=SEED+1),'dqn':DQNAgent('dqn',seed=SEED+2),'double_dqn':DQNAgent('double_dqn',seed=SEED+3)}
    for name,a in agents.items():
        path=os.path.join(MODELS_DIR,name+('.npz' if name in ('q_learning','sarsa') else '.pt'))
        if not os.path.exists(path): raise FileNotFoundError(f'Missing {path}. Run python train.py first.')
        a.load(path); a.epsilon=0.
    return agents

def main(n_episodes=100):
    os.makedirs(RESULTS_DIR,exist_ok=True); agents=load_agents(); rows=[]
    print(f'Evaluating each agent for {n_episodes} episodes...')
    for name,agent in agents.items():
        env=SnakeEnv(seed=SEED+1000,max_steps=MAX_STEPS_PER_EPISODE)
        for ep in range(1,n_episodes+1): rows.append({'algorithm':name,'episode':ep,**run_episode(env,agent,name,False)})
        subset=[r for r in rows if r['algorithm']==name]
        print(f'{name:12s} mean score={np.mean([r["score"] for r in subset]):.3f} | mean reward={np.mean([r["reward"] for r in subset]):.3f} | mean steps={np.mean([r["steps"] for r in subset]):.1f}')
    with open(os.path.join(RESULTS_DIR,'evaluation_metrics.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['algorithm','episode','reward','score','steps','epsilon']); w.writeheader(); w.writerows(rows)
    names=list(agents); groups=[[r['score'] for r in rows if r['algorithm']==n] for n in names]
    train_path=os.path.join(RESULTS_DIR,'training_metrics.csv')
    if os.path.exists(train_path):
        training={}
        with open(train_path,newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f): training.setdefault(r['algorithm'],[]).append((int(r['episode']),float(r['score'])))
        plt.figure(figsize=(10,5))
        for name,series in training.items():
            xs,ys=zip(*series); win=min(50,max(5,len(ys)//20)); sm=np.convolve(ys,np.ones(win)/win,mode='valid'); plt.plot(xs[win-1:],sm,label=name)
        plt.title('Snake learning curves (moving-average score)'); plt.xlabel('Training episode'); plt.ylabel('Score'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,'learning_curves.png'),dpi=160); plt.close()
    plt.figure(figsize=(9,5)); plt.boxplot(groups,tick_labels=names); plt.title(f'Evaluation score distribution ({n_episodes} episodes each)'); plt.ylabel('Food collected'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,'evaluation_boxplot.png'),dpi=160); plt.close()
    plt.figure(figsize=(9,5)); plt.bar(names,[np.mean(g) for g in groups],yerr=[np.std(g) for g in groups],capsize=5); plt.title('Mean evaluation score ± standard deviation'); plt.ylabel('Food collected'); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,'evaluation_comparison.png'),dpi=160); plt.close()
    print('Saved results/evaluation_metrics.csv and comparison plots in results/.')
if __name__=='__main__': main()
