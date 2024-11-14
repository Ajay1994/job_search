import torch
import torch.nn as nn
import time
import torch.nn.utils.parametrize as parameterize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TestModel(nn.Module):
    def __init__(self, hidden_dim_1 = 1000, hidden_dim_2 = 2000):
        super().__init__()

        self.linear1 = nn.Linear(28*28, hidden_dim_1)
        self.linear2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.linear3 = nn.Linear(hidden_dim_2, 10)

        self.relu = nn.ReLU()

    def forward(self, img):
        input = img.view(-1, 28 * 28)
        x = self.relu(self.linear1(input))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))

        return x
    
net = TestModel().to(device)
print("-*-"*10)
total_parameters = 0
for name, module in net.named_modules():
    if isinstance(module, nn.Linear):
        total_parameters += module.weight.nelement()
        total_parameters += module.bias.nelement()
print(f'Original Model Total number of parameters: {total_parameters:,}')

class LoRAParameterization(nn.Module):
    def __init__(self, feature_in, feature_out, rank = 1, alpha = 1, device="cpu"):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, feature_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((feature_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean = 0, std = 1)

        self.scale = alpha/rank
        self.enabled = True

    def forward(self, original_weight):
        if self.enabled:
            return original_weight + (self.lora_B @ self.lora_A).view(original_weight.shape) * self.scale
        else:
            return original_weight
        
def lora_parameterization(layer, device, rank = 1, alpha = 1):
    feature_in, feature_out = layer.weight.shape
    return LoRAParameterization(feature_in, feature_out, rank, alpha, device)

for name, module in net.named_modules():
    if isinstance(module, nn.Linear):
        parameterize.register_parametrization(module, "weight", lora_parameterization(module, device))

print("-*-"*10)
print(net)
total_parameters_lora, total_parameters = 0, 0
for name, module in net.named_modules():
    if isinstance(module, nn.Linear):
        lora_params = module.parametrizations["weight"][0].lora_A.nelement() + module.parametrizations["weight"][0].lora_B.nelement()
        total_parameters += lora_params
        total_parameters += module.weight.nelement()
        total_parameters += module.bias.nelement()

        total_parameters_lora += lora_params
print(f'LoRA Model Total number of parameters: {total_parameters_lora:,} || {total_parameters:,}')

enabled = True
def enable_disable_lora(enabled=True):
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            module.parametrizations["weight"][0].enabled = enabled

"""
-*--*--*--*--*--*--*--*--*--*-
Original Model Total number of parameters: 2,807,010
-*--*--*--*--*--*--*--*--*--*-
TestModel(
  (linear1): ParametrizedLinear(
    in_features=784, out_features=1000, bias=True
    (parametrizations): ModuleDict(
      (weight): ParametrizationList(
        (0): LoRAParameterization()
      )
    )
  )
  (linear2): ParametrizedLinear(
    in_features=1000, out_features=2000, bias=True
    (parametrizations): ModuleDict(
      (weight): ParametrizationList(
        (0): LoRAParameterization()
      )
    )
  )
  (linear3): ParametrizedLinear(
    in_features=2000, out_features=10, bias=True
    (parametrizations): ModuleDict(
      (weight): ParametrizationList(
        (0): LoRAParameterization()
      )
    )
  )
  (relu): ReLU()
)
LoRA Model Total number of parameters: 6,794 || 2,813,804
"""