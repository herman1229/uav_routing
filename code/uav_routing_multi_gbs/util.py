import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, plotting will be disabled")

def smooth(data, sm=50):
    # data: [ [list1], ... ]，这里只用到第一个list
    data = data[0]
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - sm + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed

def PlotReward(length, ReturnList, labelList, env_name):
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(12, 6))
    for i, returns in enumerate(ReturnList):
        plt.plot(range(length), returns, label=labelList[i])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Reward Curve: {env_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{env_name}_rewards.png', dpi=300, bbox_inches='tight')
    plt.show(block=True)
