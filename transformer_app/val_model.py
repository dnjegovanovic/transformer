from pathlib import Path

from modules.TransformerModule import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model_ckpt_file = Path(
        r"D:\ML_AI_DL_Projects\projects_repo\transformer\test_dir\test_01\version_1\checkpoints\epoch=19-step=36380.ckpt"
    )

    model = TransformerModule.load_from_checkpoint(model_ckpt_file)
    model.to(device)
    model.test_on_validation()
# end main
