# -*- coding: utf-8 -*-
import torch
import visdom
import random
import time
import numpy as np
from datetime import datetime


__all__ = [
    'bcolors',
    'setup_seed', 
    'Visualizer', 
    'time_delta',
    ]

class bcolors:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    gray = '\033[37m'
    bold = '\033[1m'
    underline = '\033[4m'
    flicker = '\033[5m'
    end = '\033[0m'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Visualizer(object):
    """
    封装visdom的基本操作
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env="default", **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        """
        一次img_many
        @params d: dict (name, value) i.e. ('image', img_tensor)
        """
        for k, v in d.items():
            self.img(k, v)

    def plot(self, win, name, y, **kwargs):
        """
        self.plot('loss', 1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=win,
            name=name,
            opts=dict(
                title=win,
                showlegend=False,  # 显示网格
                xlabel='x1',  # x轴标签
                ylabel='y1',  # y轴标签
                fillarea=False,  # 曲线下阴影覆盖
                width=2400,  # 画布宽
                height=350,  # 画布高
            ),
            update=None if x == 0 else 'append',
            **kwargs
        )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img, torch.Tensor(64,64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)

        !!! don't ~~self.img('input_imgs', t.Tensor(100, 64, 64), nrows=10)~~ !!!
        """
        self.vis.images(img_.cpu().numpy(), win=name,
                        opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        """
        return getattr(self.vis, name)


def time_delta(t: datetime):
    """ 计算现在到指定时间的间隔

    Parameters
    ----------
    t: datatime
        开始时间

    Returns
    -------
    delta_time: str
        时间间隔
    """
    dt = datetime.now() - t
    hours = dt.seconds//3600
    minutes = (dt.seconds-hours*3600) // 60
    seconds = dt.seconds % 60
    return f'{hours:02}:{minutes:02}:{seconds:02}'