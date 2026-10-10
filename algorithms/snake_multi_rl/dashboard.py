"""Print a compact comparison dashboard from evaluation_metrics.csv."""
import os,csv,statistics
from config import RESULTS_DIR

def main():
    path=os.path.join(RESULTS_DIR,'evaluation_metrics.csv')
    if not os.path.exists(path): print('Run python train.py and python evaluate.py first.'); return
    groups={}
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f): groups.setdefault(r['algorithm'],[]).append(r)
    print('\n=== Snake RL Evaluation Dashboard ===')
    print(f'{"Algorithm":<14} {"Mean score":>12} {"Std score":>12} {"Mean reward":>14} {"Mean steps":>12}')
    print('-'*68)
    for name,rs in groups.items():
        scores=[float(r['score']) for r in rs]; rewards=[float(r['reward']) for r in rs]; steps=[float(r['steps']) for r in rs]
        print(f'{name:<14} {statistics.mean(scores):>12.3f} {statistics.pstdev(scores):>12.3f} {statistics.mean(rewards):>14.3f} {statistics.mean(steps):>12.1f}')
    print('\nOpen these generated plots:')
    for fn in ('learning_curves.png','evaluation_boxplot.png','evaluation_comparison.png'): print(' - '+os.path.join(RESULTS_DIR,fn))
if __name__=='__main__': main()
