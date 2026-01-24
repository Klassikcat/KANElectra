import importlib
import os
from pathlib import Path
from datetime import timedelta
from typing import Literal

import torch
from torch import Tensor
import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint


class PretrainingCheckpoint(ModelCheckpoint):
    def __init__(self) -> None:
        super().__init__()  # TODO: Add a checkpointer for generator and discriminator


class OnnxCompiler(ModelCheckpoint):
    def __init__(
        self,
        dirpath: _PATH | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: Literal[True, False, "link"] | None = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
        save_onnx: bool = True,
        save_tensorrt_engine: bool = False,
        min_shape: tuple[int, int] = (1, 512),
        opt_shape: tuple[int, int] = (16, 512),
        max_shape: tuple[int, int] = (32, 512),
        save_mixed_precision: bool = True,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        self.min_shape = min_shape
        self.opt_shape = opt_shape
        self.max_shape = max_shape
        self.mixed_precision = save_mixed_precision
        dirpath_value = Path(dirpath) if dirpath is not None else Path(".")
        filename_value = filename or "model"
        self.save_dir = dirpath_value / filename_value  # TODO: find out optimial name
        if not save_onnx:
            assert (
                save_tensorrt_engine
            ), "save_tensorrt_engine must be True if save_onnx is False. if you don't want to save neither .onnx and .engine, you should use ModelCheckpoint instead."
        self.save_onnx = save_onnx
        if save_tensorrt_engine:
            trt = importlib.import_module("tensorrt")
            self.trt_logger = trt.Logger(min_severity=trt.Logger.WARNING)
            self.builder = trt.Builder(self.trt_logger)
            self.network = self.builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
        else:
            self.builder = None
            self.network = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_train_epoch_end(trainer, pl_module)
        monitor_candidates = self._monitor_candidates(trainer)
        if (
            self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        ):
            self._save_into_onnx(trainer, monitor_candidates)
            if self.builder:
                self._save_tensorRT_engine(trainer, monitor_candidates)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        super().on_validation_epoch_end(trainer, pl_module)
        monitor_candidates = self._monitor_candidates(trainer)
        if (
            self._every_n_epochs >= 1
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        ):
            onnx_save_path = self._save_into_onnx(trainer, monitor_candidates)
            if self.builder:
                self._save_tensorRT_engine(trainer, monitor_candidates, onnx_save_path)

    def _save_into_onnx(
        self, trainer: pl.Trainer, monitor_candidates: dict[str, Tensor]
    ) -> str:
        if trainer.model is None:
            raise RuntimeError("Trainer model is not initialized.")
        input_ids = torch.randint(0, 10000, (8, 512))
        attention_masks = torch.ones((8, 512), dtype=torch.long)
        token_type_ids = torch.randint(0, 1, (8, 512))

        trainer.model.eval()
        with torch.no_grad():
            _ = trainer.model(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )

        dynamic_axes = {
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "outputs": {0: "batch_size"},
        }

        onnx_save_path = str(self.save_dir) + ".onnx"
        torch.onnx.export(
            trainer.model,  # 학습된 모델 인스턴스
            (input_ids, attention_masks, token_type_ids),  # input args
            onnx_save_path,  # 저장 경로
            export_params=True,  # 모델의 학습된 파라미터를 저장할 것인지
            opset_version=17,  # 사용할 onnx의 버전
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["outputs"],
            dynamic_axes=dynamic_axes,
        )
        return onnx_save_path

    def _save_tensorRT_engine(
        self,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, Tensor],
        onnx_save_path: str | None = None,
    ) -> None:
        if self.builder is None or self.network is None:
            raise RuntimeError("TensorRT builder is not initialized.")
        trt = importlib.import_module("tensorrt")

        parser = trt.OnnxParser(self.network, self.trt_logger)
        parser.parse_from_file(str(self.save_dir) + ".onnx")
        config = self.builder.create_builder_config()
        profile = self.builder.create_optimization_profile()
        profile.set_shape(
            "input_ids",
            (self.min_shape[0], self.min_shape[1]),
            (self.opt_shape[0], self.opt_shape[1]),
            (self.max_shape[0], self.max_shape[1]),
        )
        profile.set_shape(
            "attention_mask",
            (self.min_shape[0], self.min_shape[1]),
            (self.opt_shape[0], self.opt_shape[1]),
            (self.max_shape[0], self.max_shape[1]),
        )
        profile.set_shape(
            "token_type_ids",
            (self.min_shape[0], self.min_shape[1]),
            (self.opt_shape[0], self.opt_shape[1]),
            (self.max_shape[0], self.max_shape[1]),
        )
        config.add_optimization_profile(profile)
        if self.mixed_precision:
            config.set_flag(trt.BuilderFlag.FP16)
        serialized_engine = self.builder.build_serialized_network(self.network, config)
        with open(f"{str(self.save_dir)}.plan", "wb") as engine:
            engine.write(serialized_engine)
        if not self.save_onnx:
            os.remove(f"{str(self.save_dir)}.onnx")
