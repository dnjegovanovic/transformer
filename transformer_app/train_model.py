import pytorch_lightning.loggers as loggers
from config.core import config
from modules.TransformerModule import *

if __name__ == "__main__":
    save_dir = config.app_config.save_file
    print(f"Save model to:{save_dir}")
    logger = loggers.TensorBoardLogger(
        save_dir=save_dir, name="test_01", version=1, log_graph=True
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.5f}",
        save_top_k=2,
        mode="min",
        save_last=True,
    )

    train = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=logger,
        max_epochs=config.model_transformer.TR_model["epochs"],
        log_every_n_steps=5,
        callbacks=[checkpoint_callback]
    )

    model = TransformerModule(**vars(config.model_transformer))
    train.fit(model, ckpt_path=config.model_transformer.TR_model["ckpt_file"])
# end main
