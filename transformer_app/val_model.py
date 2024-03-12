from pathlib import Path

from modules.TransformerModule import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model_ckpt_file = Path(
        r"test_dir\test_01\version_1\checkpoints\model-epoch=09-val_loss=4.56244.ckpt"
    )

    model = TransformerModule.load_from_checkpoint(model_ckpt_file, batch_size=1)
    model.to(device)
    model.test_on_validation()
# end main
