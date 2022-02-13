import matplotlib.pyplot as plt
import numpy as np
import torch

from training.LabelSmoothing import LabelSmoothing


def loss(x):
    crit = LabelSmoothing(5, 0, 0.1)
    d = x + 3 * 1
    predict = torch.FloatTensor(
        [
            [0, x / d, 1 / d, 1 / d, 1 / d],
        ]
    )
    # print(predict)
    t1 = torch.autograd.Variable(predict.log())
    t2 = torch.autograd.Variable(torch.LongTensor([1]))
    c = crit(t1, t2).item()
    return c


def test_loss_penalization():

    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.savefig("label_loss_penalizing.png")
    plt.close()
