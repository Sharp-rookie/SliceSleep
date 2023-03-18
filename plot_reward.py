import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('log/ue_5/PPO/log.csv')
plt.figure()
plt.plot(df['episode'], df['reward1'], label='slice1')
plt.plot(df['episode'], df['reward2'], label='slice2')
plt.plot(df['episode'], df['reward3'], label='slice3')
plt.legend()
plt.savefig('reward.jpg', dpi=300)