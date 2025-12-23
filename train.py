from model.rebuild_trainer import RebuildTrainer
from model.pred_trainer import PredTrainer
from config import *


def train_rebuild_model(config: dict) -> None:
    rebuild_trianer = RebuildTrainer(config)
    rebuild_trianer.train()
    rebuild_trianer.test()


def train_pred_model(config: dict) -> None:
    pred_trianer = PredTrainer(config)
    pred_trianer.train()
    pred_trianer.test()


if __name__ == "__main__":
    config = load_config("config.yaml")
    train_rebuild_model(config)
    train_pred_model(config)
