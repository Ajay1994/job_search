import torch
import torch.nn as nn

activation = {}
def getActivation(module_name):
    def hook(model, input, output):
        activation[module_name] = output.detach()
    return hook

class FeedForward(nn.Module):
    def __init__(self, dim_in = 1024, dim_hidden = 2048, dim_out = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_hidden, bias=False)] +
            [nn.Linear(dim_hidden, dim_hidden, bias=False) for _ in range(0, 5)] +
            [nn.Linear(dim_hidden, dim_out, bias=False)]
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x

device = torch.device("cuda:0")
input_x = torch.randn(8, 1024, device=device)
input_y = torch.randn(8, 3, device=device)
model = FeedForward()
model.layers[0].register_forward_hook(getActivation('layer-0-activation'))
print(model.to(device))

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00001)

for i in range(0, 500):
    
    output = model(input_x)
    loss = (output - input_y).pow(2).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0: print(f"Loss = {loss.item():.3f}")

print(activation, activation['layer-0-activation'].shape)



