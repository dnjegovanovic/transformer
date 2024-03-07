from transformer_app.modules.TransformerModule import *
from transformer_app.config.core import config

import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers

if __name__ == "__main__":
    save_dir = config.app_config.save_file
    print(f"Save model to:{save_dir}")
    logger = loggers.TensorBoardLogger(
        save_dir=save_dir, name="test_01", version=1, log_graph=True
    )

    train = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=logger,
        check_val_every_n_epoch=10,
        max_epochs=config.model_transformer.TR_model["epochs"],
        log_every_n_steps=5,
    )
    
    model = TransformerModule(**vars(config.model_transformer))
    train.fit(model)
# end main
