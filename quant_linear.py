from hashlib import new
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Callable
from enum import Enum


class QuantizationMode(Enum):
    one_bit = 1
    two_bit = 2


class BitNetLinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        quantization_mode: QuantizationMode = QuantizationMode.two_bit,
    ):
        super(BitNetLinearLayer, self).__init__()
        self.binary_layer = True
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = (
            nn.Parameter(torch.Tensor(out_features)) if bias is not None else None
        )
        self.quantization_mode = quantization_mode

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def compute_adjustment_factor(self, input_tensor: torch.Tensor):
        absmean_weight = torch.mean(torch.abs(input_tensor))
        adjustment_factor = 1e-4 + absmean_weight * 2 + 1e-4
        return adjustment_factor

    def compute_2bit_quantized_tensor(self, input_tensor: torch.Tensor):
        twobit_matrix = torch.clip(input=torch.round(input_tensor), min=-1, max=1)
        return twobit_matrix

    def compute_1bit_quantized_tensor(self, input_tensor: torch.Tensor):
        return torch.sign(input_tensor)

    def compute_quantized_tensor(self, input_tensor: torch.Tensor):
        if self.quantization_mode == QuantizationMode.two_bit:
            return self.compute_2bit_quantized_tensor(input_tensor)
        else:
            return self.compute_1bit_quantized_tensor(input_tensor)

    def compute_commitment_loss(
        self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss
    ):
        adjustment_factor = self.compute_adjustment_factor(self.weight)
        adjusted_weight = self.weight / adjustment_factor
        quantized_weight = self.compute_quantized_weight(adjusted_weight)

        return loss_fn(adjusted_weight, quantized_weight.detach())

    def forward(self, x):
        weight_adjustment_factor = self.compute_adjustment_factor(self.weight)
        adjusted_weight = self.weight / weight_adjustment_factor
        input_adjustment_factor = 127.0
        adjusted_input = x / input_adjustment_factor

        quantized_weight = self.compute_quantized_tensor(adjusted_weight)
        quantized_input = torch.clip(torch.round(adjusted_input), min=-1, max=1)

        if self.training:
            quantized_weight = (
                adjusted_weight + (quantized_weight - adjusted_weight).detach()
            )

            quantized_input = (
                adjusted_input + (quantized_input - adjusted_input).detach()
            )

        output = (
            weight_adjustment_factor
            * input_adjustment_factor
            * adjusted_input
            @ adjusted_weight.t()
        )

        if self.bias is not None:
            output += self.bias
        return output


import copy


def create_quantized_copy_of_model(
    input_model: nn.Module, quantization_mode: QuantizationMode
):
    model_copy = copy.deepcopy(input_model)
    hash_table = {n: m for n, m in model_copy.named_modules()}

    for key in list(hash_table.keys()):
        if isinstance(hash_table[key], nn.Linear):
            new_module = BitNetLinearLayer(
                in_features=hash_table[key].in_features,
                out_features=hash_table[key].out_features,
                bias=hash_table[key].bias is not None,
                quantization_mode=quantization_mode,
            )
            name_chain = key.split(".")
            parent_module_attr_name = ".".join(name_chain[:-1])
            parent_module = hash_table[parent_module_attr_name]
            setattr(parent_module, name_chain[-1], new_module)
    for n, m in model_copy.named_modules():
        assert not isinstance(m, nn.Linear)
    return model_copy
