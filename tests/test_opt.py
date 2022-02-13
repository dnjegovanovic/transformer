import matplotlib.pyplot as plt
import numpy as np

from model.Optimizer import Optimizer


def test_optimizer():
    opts = [
        Optimizer(512, 1, 4000, None),
        Optimizer(512, 1, 8000, None),
        Optimizer(256, 1, 4000, None),
    ]
    plt.plot(
        np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)]
    )
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.savefig("test_opt.png")
    plt.close()
