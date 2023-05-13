import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('log/ue_5/PPO/log.csv')
plt.figure()
plt.plot(df['episode'], df['reward1'], label='slice1')
plt.plot(df['episode'], df['reward2'], label='slice2')
plt.plot(df['episode'], df['reward3'], label='slice3')
plt.legend()
plt.savefig('reward.jpg', dpi=300)


for i in range(1,4):
    df = pd.read_csv(f'log/ue_5/PPO/log_pool{i}.csv')

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(df['offset'], df['action'], df['reward'], s=3)
    ax.set_xlabel('offset')
    ax.set_ylabel('action')
    ax.view_init(elev=25., azim=25.)
    plt.savefig(f'log/ue_5/PPO/log_pool{i}.jpg', dpi=300)
    
    # corr = df.corr('pearson')
    # corr.to_csv(f'log/ue_5/PPO/log_pool{i}_corr.csv')
    # f, ax= plt.subplots()
    # sns.heatmap(corr, linewidths = 0.05, ax = ax)
    # ax.set_title('Correlation between features')
    # f.savefig(f'log/ue_5/PPO/log_pool{i}_corr.jpg', dpi=300, bbox_inches='tight')