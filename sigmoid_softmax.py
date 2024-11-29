import torch
import torch.nn as nn
def _sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class MySigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return _sigmoid(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_local = _sigmoid(input) * (1 - _sigmoid(input))
        grad_input = grad_local * grad_output.clone()
        return grad_input
    

weight = nn.Parameter(torch.randn(5, 5))
input = torch.randn(5)
output = MySigmoid.apply(input @ weight)
bce_loss = torch.nn.BCELoss()
loss = bce_loss(output, torch.Tensor([0, 0, 0, 0, 0]))
loss.backward()


print(input.shape, output.shape, input, output, loss, weight.grad)