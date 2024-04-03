from datasets import load_dataset
import lightning as L
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import ViTModel, ViTForImageClassification


from quant_linear import (
    create_quantized_copy_of_model,
    QuantizationMode,
)

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

config = ViTConfig(
    hidden_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    intermediate_size=256,
    hidden_act="gelu",
    image_size=32,
    patch_size=4,
    num_labels=100,
    num_channels=3,
)


class ViTImageClassifier(L.LightningModule):
    def __init__(self, config: ViTConfig, lr=1e-3):
        super().__init__()
        self.model = ViTForImageClassification(config)
        self.config = config
        self.lr = lr

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        argmax = output.logits.argmax(dim=1)
        accuracy = (argmax == batch["labels"]).float().mean()
        self.log_dict(
            {
                "tl": loss.item(),
                "ta": accuracy.item(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self(batch)
            loss = output.loss
            argmax = output.logits.argmax(dim=1)
            accuracy = (argmax == batch["labels"]).float().mean()

        self.log_dict(
            {
                "vl": loss.item(),
                "va": accuracy.item(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


dataset = load_dataset("cifar100")

image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

processed_dataset = dataset.map(
    lambda x: {"pixel_values": image_transforms(x["img"]), "labels": x["fine_label"]}
)
processed_dataset = processed_dataset.remove_columns(["fine_label", "img"])
processed_dataset.set_format("torch", columns=["pixel_values", "labels"])


train_dataloader = DataLoader(processed_dataset["train"], batch_size=128)
eval_dataloader = DataLoader(processed_dataset["test"], batch_size=128)

normal_model = ViTImageClassifier(config)
one_bit_quantized_model = create_quantized_copy_of_model(
    normal_model, quantization_mode=QuantizationMode.one_bit
)
two_bit_quantized_model = create_quantized_copy_of_model(
    normal_model, quantization_mode=QuantizationMode.two_bit
)

from lightning.pytorch.loggers import WandbLogger

choice = input("Enter 1,2,3:")
if int(choice) == 1:
    normal_logger = WandbLogger(project="BitNet_v2", name="normal_cifar100")
    normal_trainer = L.Trainer(
        max_epochs=10,
        logger=normal_logger,
    )
    normal_trainer.fit(
        normal_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )
if int(choice) == 2:
    one_bit_logger = WandbLogger(project="BitNet_v2", name="one_bit_cifar100")
    one_bit_trainer = L.Trainer(
        max_epochs=10,
        logger=one_bit_logger,
    )
    one_bit_quantized_model.lr = 1e-4
    one_bit_trainer.fit(
        one_bit_quantized_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )

if int(choice) == 3:
    two_bit_logger = WandbLogger(project="BitNet_v2", name="two_bit_cifar100")
    two_bit_trainer = L.Trainer(
        max_epochs=10,
        logger=two_bit_logger,
    )
    two_bit_quantized_model.lr = 1e-4
    two_bit_trainer.fit(
        two_bit_quantized_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )
