import torch

from app.utils.visualize_mask import subsequent_mask


class Batch:
    def __init__(self, src, trg, padding=0):
        self.src = src
        self.trg = trg

        self.src_mask = (src != padding).unsqueeze(-2)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]

            self.trg_mask = self.make_mask(self.trg, padding)

            self.ntokens = (self.trg_y != padding).data.sum()

    @staticmethod
    def make_mask(trg, padding):
        "Create a mask to hide padding and future words."
        trg_mask = (trg != padding).unsqueeze(-2)
        trg_mask = trg_mask & torch.autograd.Variable(
            subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
        )
        return trg_mask
