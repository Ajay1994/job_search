import torch
import torch.nn as nn
import torch.functional as F

class CustomLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(self.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdims = True)
        var = x.var(dim = -1, keepdims = True)
        x_norm = (x - mean)/torch.sqrt(var + self.eps)
        return self.alpha * x_norm + self.beta
    
x = torch.randn(2, 4)
custom_layer_norm = CustomLayerNorm(4)
normalized_x = custom_layer_norm(x)
print(normalized_x)

##############################################

class CustomRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x) 
        return x * torch.rsqrt(
            x.pow(2).mean(dim = -1, keepdim = True) 
            + self.eps
            )
    
    def forward(self, x: torch.Tensor):
        # x.shape = [B, seq_len, dim] 
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
x = torch.randn(2, 4)
custom_rms_norm = CustomRMSNorm(4)
normalized_x = custom_rms_norm(x)
print(normalized_x)