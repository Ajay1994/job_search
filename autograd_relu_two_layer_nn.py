# Ref: https://github.com/jcjohnson/pytorch-examples/blob/master/autograd/two_layer_net_custom_function.py

import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        In forward pass, we receive context object and Tensor containing the i/p;
        we must return Tensor object containing the output, and we can use 
        context object to cache objects for use in the backward pass.
        """
        ctx.save_for_backward(x)
        return x.clamp(min = 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In backward pass, we receive context object and a Tensor contraining 
        the gradient of loss wrt. o/p produced during the forward pass.
        We can retrive cached data from context object, and must compute
        and return gradient of loss wrt. input to the forward function.
        """
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x
    

device = torch.device("cuda:0")
N, input_dim, hidden_dim, output_dim = 64, 1024, 256, 10
x = torch.randn(N, input_dim, device = device)
y = torch.randn(N, output_dim, device = device)
W1 =  torch.randn(input_dim, hidden_dim, device=device, requires_grad=True)
W2 =  torch.randn(hidden_dim, output_dim, device=device, requires_grad=True)

lr = 1e-6

for i in range(500):
    output = MyReLU.apply(x @ W1) @ W2

    loss = (y - output).pow(2).sum()
    print(i, loss.item())

    loss.backward()

    with torch.no_grad():
        W1 -= lr * W1.grad
        W2 -= lr * W2.grad

        W1.grad.zero_()
        W2.grad.zero_()

