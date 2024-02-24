"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from pflow import utils
from pflow.utils.utils import plot_tensor
from pflow.models.components import commons

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            print(self.ckpt_loaded_epoch - 1)
            if hasattr(self, "ckpt_loaded_epoch"):
                scheduler.last_epoch = self.ckpt_loaded_epoch - 1
            else:
                scheduler.last_epoch = -1
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "interval": self.hparams.scheduler.lightning_args.interval,
                    # "frequency": self.hparams.scheduler.lightning_args.frequency,
                    # "name": "learning_rate",
                    "monitor": "val_loss",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        # prompt_spec = batch["prompt_spec"]
        # prompt_lengths = batch["prompt_lengths"]
        # prompt_slice, ids_slice = commons.rand_slice_segments(
        #                 prompt_spec, 
        #                 prompt_lengths, 
        #                 self.prompt_size
        #             )
        prompt_slice = None
        dur_loss, prior_loss, diff_loss, attn = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            prompt=prompt_slice,
        )
        return ({
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
            },
            {
            "attn": attn
            }
            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict, attn_dict = self.get_losses(batch)
        # print(batch.keys())
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        attn = attn_dict["attn"][0]
        self.logger.experiment.add_image(
                    f"train/alignment",
                    plot_tensor(attn.cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict, attn_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        attn = attn_dict["attn"][0]
        self.logger.experiment.add_image(
                    f"val/alignment",
                    plot_tensor(attn.cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))

            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                y = one_batch["y"][i].unsqueeze(0).to(self.device)
                y_lengths = one_batch["y_lengths"][i].unsqueeze(0).to(self.device)
                # prompt = one_batch["prompt_spec"][i].unsqueeze(0).to(self.device)
                # prompt_lengths = one_batch["prompt_lengths"][i].unsqueeze(0).to(self.device)
                prompt = y
                prompt_lengths = y_lengths
                prompt_slice, ids_slice = commons.rand_slice_segments(
                        prompt, prompt_lengths, self.prompt_size
                    )
                output = self.synthesise(x[:, :x_lengths], x_lengths, prompt=prompt_slice, n_timesteps=10, guidance_scale=0.0)
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
