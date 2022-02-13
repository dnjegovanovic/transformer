import matplotlib.pyplot as plt
import numpy as np
import torch

from training.LabelSmoothing import LabelSmoothing


def test_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [[0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]]
    )
    v = crit(
        torch.autograd.Variable(predict.log()),
        torch.autograd.Variable(torch.LongTensor([2, 1, 0])),
    )

    # Show the target distributions expected by the system.
    plt.matshow(crit.true_dist)
    plt.savefig("label_smoothing_test.png")
    plt.close()
