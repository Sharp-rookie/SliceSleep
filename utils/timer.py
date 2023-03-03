# -*- coding: utf-8 -*-
from time import sleep, time

__all__ = [
    'RepeatedTimer',
    ]


class RepeatedTimer(object):
    """定时器工具，用于定时执行程序，本工程中用于每个TTI执行simulate()
    """
    def __init__(self, length, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.is_running = False
        self.args = args
        self.kwargs = kwargs
        self.start(length)

    def start(self, length):
        start = int(time() * 1000)
        i = 0
        if not self.is_running:
            self.is_running = True

            first_execute = 0
            now = 0
            while True:
                if not self.is_running:
                    break

                i += 1
                last = now
                now = int(time() * 1000)
                sleep_ms = i * self.interval - (now - start)

                # 定时关闭，共length秒
                if now-start >= length*1000:
                    # add breakpoint here
                    self.function(*self.args, **self.kwargs)
                    print(f'run {length}s over!')
                    exit()

                if sleep_ms < 0:
                    if first_execute == 0:
                        first_execute += 1
                    else:
                        # print(f"Time Excess! {now - last}ms")
                        pass
                    self.function(*self.args, **self.kwargs)
                else:
                    sleep(sleep_ms / 1000.0)
                    self.function(*self.args, **self.kwargs)

    def stop(self):
        self.is_running = False
