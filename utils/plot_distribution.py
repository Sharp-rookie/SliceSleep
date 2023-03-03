from math import trunc
from matplotlib.pyplot import subplot
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import uniform, poisson, expon, pareto

u = uniform(loc=0, scale=100) # loc~scale内均匀分布
p = poisson(mu=100) # 均值为mu的泊松分布
e = expon(loc=0, scale=100) # 从loc开始，均值为loc+scale的指数分布
pp = pareto(b=300) # 搞不懂

plt.figure(dpi=250)
plt.subplots_adjust(wspace=0.35,hspace=0.35)

plt.subplot(2,2,1)
plt.title('uniform')
plt.hist(u.rvs(30000), bins=50)

plt.subplot(2,2,2)
plt.title('poisson')
plt.hist(p.rvs(30000), bins=50)

plt.subplot(2,2,3)
plt.title('expon')
plt.hist(e.rvs(30000), bins=50)

plt.subplot(2,2,4)
plt.title('pareto')
plt.hist(pp.rvs(30000), bins=50)

plt.savefig('plot_distribution.jpg')