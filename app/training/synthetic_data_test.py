import numpy as np
import torch

from app.model.Optimizer import Optimizer
from app.model.Transformer import Transformer
from app.training.BatchesAndMasking import Batch
from app.training.LabelSmoothing import LabelSmoothing
from app.training.train import run_epoch


def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""

    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = torch.autograd.Variable(data, requires_grad=False)
        tgt = torch.autograd.Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


def train_synth_test(num_of_epoch=10):
    # Train the simple copy task.
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    transformer = Transformer(V, V, number_of_layers=2)
    transformer_model = transformer.make_model()
    model_opt = Optimizer(
        transformer_model.src_embed[0].d_model,
        1,
        400,
        torch.optim.Adam(
            transformer_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        ),
    )

    for epoch in range(num_of_epoch):
        transformer_model.train()
        run_epoch(
            data_gen(V, 30, 20),
            transformer_model,
            SimpleLossCompute(transformer_model.generator, criterion, model_opt),
        )
        transformer_model.eval()
        print(
            run_epoch(
                data_gen(V, 30, 5),
                transformer_model,
                SimpleLossCompute(transformer_model.generator, criterion, None),
            )
        )
