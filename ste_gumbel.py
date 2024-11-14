import math
import torch
import torch.nn as nn
class STE_Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        #Apply a non-differentiable operation
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        #Gradient from next layers reached here. Simply pass it through this operation
        return grad_output
    
# weight = nn.Parameter(torch.randn(5, 6))
# input = torch.randn(1, 5)
# output = torch.sum(input @ weight)
# output.backward()
# print(output, weight.grad)
"""
tensor(8.2291, grad_fn=<SumBackward0>) 
tensor([[ 0.8452,  0.8452,  0.8452,  0.8452,  0.8452,  0.8452],
        [-0.2425, -0.2425, -0.2425, -0.2425, -0.2425, -0.2425],
        [ 0.1117,  0.1117,  0.1117,  0.1117,  0.1117,  0.1117],
        [-1.2184, -1.2184, -1.2184, -1.2184, -1.2184, -1.2184],
        [-1.1750, -1.1750, -1.1750, -1.1750, -1.1750, -1.1750]])
"""


weight = nn.Parameter(torch.randn(5, 6))
input = torch.randn(1, 5)
output = input @ weight
output = torch.sum(STE_Round.apply(output))
output.backward()
print(output, weight.grad)

"""
tensor(-5., grad_fn=<SumBackward0>) 
tensor([[-0.7003, -0.7003, -0.7003, -0.7003, -0.7003, -0.7003],
        [ 2.1890,  2.1890,  2.1890,  2.1890,  2.1890,  2.1890],
        [-1.8203, -1.8203, -1.8203, -1.8203, -1.8203, -1.8203],
        [ 0.1748,  0.1748,  0.1748,  0.1748,  0.1748,  0.1748],
        [ 0.9876,  0.9876,  0.9876,  0.9876,  0.9876,  0.9876]])
"""
