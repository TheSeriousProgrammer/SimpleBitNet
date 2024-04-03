import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from quant_linear import (
    create_quantized_copy_of_model,
    QuantizationMode,
)


class MnistLightning(L.LightningModule):
    def __init__(self, linear_layer_type: nn.Module = nn.Linear, lr=1e-3):
        super().__init__()
        self.linear_layer_type = linear_layer_type
        self.model = nn.Sequential(
            linear_layer_type(28 * 28, 128),
            nn.ReLU(),
            linear_layer_type(128, 10, bias=False),
            nn.ReLU(),
            linear_layer_type(10, 10),
            nn.ReLU(),
            linear_layer_type(10, 10),
        )
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        accuracy = torch.argmax(logits, 1).eq(y).float().mean()
        self.log_dict(
            {"tl": loss.item(), "ta": accuracy.item()},
            on_step=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            x = x.view(x.size(0), -1)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            accuracy = torch.argmax(logits, 1).eq(y).float().mean()
        self.log_dict(
            {"vl": loss.item(), "va": accuracy.item()}, on_step=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        return optimizer


dataset_folder = os.path.join(os.getcwd(), "data")
# setup data

train_dataset = MNIST(dataset_folder, train=True, download=True, transform=ToTensor())
test_dataset = MNIST(dataset_folder, train=False, download=True, transform=ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

print("NORMAL TRAINING")

from lightning.pytorch.loggers import WandbLogger

normal_module = MnistLightning(linear_layer_type=nn.Linear)
one_bit_quantized_module = create_quantized_copy_of_model(
    normal_module, quantization_mode=QuantizationMode.one_bit
)
two_bit_quantized_module = create_quantized_copy_of_model(
    normal_module, quantization_mode=QuantizationMode.two_bit
)

input_val = input("enter 1,2,3")
if int(input_val) == 1:
    normal_logger = WandbLogger(project="BitNet_v2", name="normal_mnist")
    normal_trainer = L.Trainer(
        max_epochs=10,
        logger=normal_logger,
    )
    normal_trainer.fit(normal_module, train_loader, test_loader)
    normal_logger.finalize(status="success")

if int(input_val) == 2:
    one_bit_logger = WandbLogger(project="BitNet_v2", name="one_bit_mnist")
    one_bit_logger.experiment.name = "one_bit_mnist"
    one_bit_quantized_module.lr = 1e-4
    one_bit_quant_trainer = L.Trainer(
        max_epochs=10,
        logger=one_bit_logger,
    )
    one_bit_quant_trainer.fit(one_bit_quantized_module, train_loader, test_loader)
    one_bit_logger.finalize(status="success")

if int(input_val) == 3:
    two_bit_logger = WandbLogger(project="BitNet_v2", name="two_bit_mnist_lr=1e-4")
    two_bit_logger.experiment.name = "two_bit_mnist_lr=1e-4"
    two_bit_quant_trainer = L.Trainer(
        max_epochs=10,
        logger=two_bit_logger,
    )
    two_bit_quantized_module.lr = 1e-4
    two_bit_quant_trainer.fit(two_bit_quantized_module, train_loader, test_loader)
    two_bit_logger.finalize(status="success")
