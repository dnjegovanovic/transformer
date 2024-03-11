from pathlib import Path

from modules.TransformerModule import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model_ckpt_file = Path(
        r"test_dir\test_03\version_1\checkpoints\epoch=19-step=36380.ckpt"
    )

    model = TransformerModule.load_from_checkpoint(model_ckpt_file, batch_size=1)
    model.to(device)
    model.test_on_validation()
# end main
