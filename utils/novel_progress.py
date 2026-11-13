"""Write novel-species training progress where the explorer can read it.

``lightning_logs/<exp>/progress.json`` is replaced after each step and after
each validation. The file is small on purpose: stage, epoch, and the species
F1 numbers, not the full log.
"""

import json
from pathlib import Path

import pytorch_lightning as pl


def _float(value):
    if value is None:
        return None
    try:
        number = float(value.detach() if hasattr(value, "detach") else value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return round(number, 6)


class NovelProgress(pl.Callback):
    """One JSON file the few-shot plan diagram polls."""

    def __init__(self, path, stage, species, stats_names, max_epochs):
        self.path = Path(path)
        self.stage = stage or "joint"
        self.species = species
        self.stats_names = list(stats_names)
        self.max_epochs = int(max_epochs)
        self.index = self.stats_names.index(species)
        self.state = {
            "stage": self.stage,
            "species": species,
            "epoch": 0,
            "max_epochs": self.max_epochs,
            "step": 0,
            "steps": None,
            "novel_f1": None,
            "old_f1": {},
        }

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.state["epoch"] = int(trainer.current_epoch) + 1
        self.state["step"] = int(batch_idx) + 1
        total = trainer.num_training_batches
        self.state["steps"] = int(total) if isinstance(total, int) and total > 0 else None
        if batch_idx % 20 == 0:
            self._write()

    def on_train_epoch_end(self, trainer, pl_module):
        self.state["epoch"] = int(trainer.current_epoch) + 1
        self._write()

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        old = {}
        for index, name in enumerate(self.stats_names):
            if index == 0 or name == "background":
                continue
            score = _float(metrics.get(f"val_f1_class_{index}"))
            if name == self.species:
                self.state["novel_f1"] = score
            else:
                old[name] = score
        self.state["old_f1"] = old
        self.state["epoch"] = int(trainer.current_epoch) + 1
        self._write()
