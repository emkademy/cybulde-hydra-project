import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


"""

Train a model on the CIFAR10 dataset

Tasks:

* Check the created data module: data_modules.CIFAR10DataModule
* Implement the model: models.CIFAR10Model
* Implement the task: tasks.CIFAR10Task
* Create an appropriate adapter (only the config file is necessary)
* Create all other necessary config files: cifar10_classification.yaml, sgd.yaml, cifar10_model, ...
* Run the training: python train.py
* Check the logs: tensorboard --logdir tb_logs
* Check the checkpoints: ls checkpoints

What to use:
1. Try different backbones (replacing them using hydra)
1. Use SGD as the optimizer
2. Use CrossEntropyLoss as the loss function
3. Use the accuracy metric

Bonus:
1. Train MNIST classifier by overriding the task in the hydra config.

"""


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(config: DictConfig) -> None:
    data_module = instantiate(config.data_module)
    task = instantiate(config.task)

    # Create the logger
    tb_logger = TensorBoardLogger("tb_logs", name="cifar10")

    # Create the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_accuracy",
        dirpath="checkpoints",
        filename="cifar10-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="max",
    )

    # Create the trainer
    trainer = instantiate(config.trainer, logger=tb_logger, callbacks=[checkpoint_callback])

    # Fit the model
    trainer.fit(task, datamodule=data_module)

    # Test the model
    trainer.test(datamodule=data_module)


# Run the train function
if __name__ == "__main__":
    train()
