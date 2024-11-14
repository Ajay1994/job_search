import pdb
import time
import math
import torch
import torch.nn as nn
from torch.nn import Linear


def _quantize_tensor_uint8(
    w, q_group_size = -1, n_bit = 8
):
    original_shape = w.shape
    if q_group_size > 0:
        assert w.nelement() % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    
    assert w.dim() == 2
    max_val = w.amax(dim = 1, keepdims = True)
    min_val = w.amin(dim = 1, keepdims = True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5)/max_int
    zeros = (-torch.round(min_val/scales)).clamp_(min_int, max_int)
    
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0
    
    w = torch.clamp(
        torch.round(w/scales) + zeros,
        min_int, max_int
    )
    w = w.reshape(original_shape).to(torch.uint8)
    return w, scales, zeros


class W8Linear(torch.autograd.Function):
    # https://fleuret.org/dlc/materials/dlc-slides-5-7-writing-an-autograd-function.pdf 
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
       
        def forward_w_float_weight(weight, x, bias):
           float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)
           (float_weight.sub_(weight.zeros)).mul_(weight.scales)
           float_weight = float_weight.reshape(weight.shape)
           return x @ float_weight.t() + bias if bias is not None else x @ float_weight.t()
       
        output = forward_w_float_weight(weight, x, bias)
        return output
    
    @staticmethod
    def backward(ctx, gradient_wrt_output):
        """
        grad_input contains the gradients wrt to the input of the layer. Similarly, the grad_output 
        contains the gradients wrt to the output of the layer. Since backprop works in reverse, 
        grad_output is what got propagated from the next layer while grad_input is what will be sent to the previous one.
        """
        x, weight, bias = ctx.saved_tensors
    
        # O = WX | grad(X) = W
        def backward_w_float_weight(weight, gradient_output):
            float_weight = weight.to(x.dtype).reshape(-1, weight.group_size)
            (float_weight.sub_(weight.zeros)).mul_(weight.scales)
            float_weight = float_weight.reshape(weight.shape)
            grad_wrt_input = gradient_output @ float_weight
            return grad_wrt_input
        
        gradient_wrt_input = backward_w_float_weight(weight, gradient_wrt_output)
        
        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = gradient_wrt_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None
        
        out_features, in_features = weight.shape 
        # Gradient Accumulation [gradient of the weight]
        if not hasattr(weight, 'float_grad'):
            weight.__setattr__('float_grad', None)
            
        if weight.float_grad is not None:
            weight.float_grad += gradient_wrt_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features)
        else:
            weight.float_grad = gradient_wrt_output.reshape(-1, out_features).t() @ x.reshape(-1, in_features)
            
        # if hasattr(weight, 'backward_hook'):
        #     weight.backward_hook(weight)
        
        return gradient_wrt_input, None, grad_bias
    
    
class QGaloreLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        num_bits = 8,
        group_size = 256,
        stochastic_round = True,
        device = None
    ) -> None:
        super().__init__()
        
        int8_weight, scales, zeros = _quantize_tensor_uint8(weight.data, q_group_size = group_size)
        torch.cuda.empty_cache()
        
        #Gradients are only possible for float
        self.weight = nn.Parameter(int8_weight, requires_grad=False).to(device)
        self.weight.__setattr__('scales', scales.to(device))
        self.weight.__setattr__('zeros', zeros.to(device))
        self.weight.__setattr__('group_size', group_size)
        self.weight.__setattr__('saved_dtype', int8_weight.dtype)
        self.weight.__setattr__('stochastic_round', stochastic_round)
        
        if not num_bits == 8: raise NotImplementedError
        
        self.bias = nn.Parameter(bias, requires_grad=True).to(device) if bias is not None else None
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Implementation of STE to calcuate forward and backward pass through unit8 layer [W8Linear is grad function for tensor]
        output = W8Linear.apply(input, self.weight, self.bias)
        return output
        


if __name__== '__main__':
    GROUP_SIZE = 256
    
    print("**** Memory check for a Single Linear Layer ****")
    fp16_linear1 = Linear(4096, 4096, bias=False).to('cuda:0').to(torch.bfloat16)
    print("Initial weight with bfloat16 memory allocated: {:.2f} MB".format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    memory_weight_float = torch.cuda.memory_allocated('cuda:0')//1024/1024
    
    input = torch.rand(1, 1024, 4096, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
    print("After Initial input with bfloat16 memory allocated: {:.2f} MB".format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    
    start = time.time()
    output = fp16_linear1(input)
    print('After forward for bfloat16 memory allocated', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    output.sum().backward()
    end = time.time()
    print("After backward pass with bfloat16 memory allocated: {:.2f} MB".format(torch.cuda.memory_allocated('cuda:0')//1024/1024))
    print("Time Taken FW + BW : {:2f} sec.".format(end-start))
    
    print("----------------------------------------------------------")
    
    int8_linear1 = QGaloreLinear(fp16_linear1.weight, None, num_bits=8, group_size=GROUP_SIZE, stochastic_round=True, device="cuda:1")
    print('Initial weight with int8 memory allocated:', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    mem_weight_int = torch.cuda.memory_allocated('cuda:1')//1024/1024
    input1 = torch.randn(1, 1024, 4096, dtype=torch.bfloat16, device='cuda:1', requires_grad=True)
    print('After initial input for bfloat16 memory allocated:', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    
    start = time.time()
    output_int8 = int8_linear1(input1)
    print('After forward for int8 memory allocated:', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    output_int8.sum().backward()
    end = time.time()
    print('After backward for int8 memory allocated:', '{:.2f} MB'.format(torch.cuda.memory_allocated('cuda:1')//1024/1024))
    print('Time for FW+BW = {:.2f} s'.format(end-start))
    print('------------------------------------')

    print('Memory saving for weight: {:.2f} MB, ratio: {:.2f}%'.format(memory_weight_float - mem_weight_int, mem_weight_int / memory_weight_float * 100))
    print('------------------------------------')