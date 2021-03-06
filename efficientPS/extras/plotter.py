from visdom import Visdom
import numpy as np


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name="main"):
        self.viz = Visdom(port=8098)
        self.env = env_name
        self.plots = {}
        self.counts = {}

    def plot(self, var_name, split_name, title_name, y):
        if var_name not in self.plots:
            self.counts[var_name] = 0
            x = self.counts[var_name]
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel="Epochs",
                    ylabel=var_name,
                ),
            )
        else:
            self.counts[var_name] += 1
            x = self.counts[var_name]
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
            )
