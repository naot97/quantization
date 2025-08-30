import torch
import torch.nn as nn
import torch.nn.functional as F

def w8_a16_forward(weight, input, scales, bias=None):
    
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    
    if bias is not None:
        output = output + bias
      
    return output

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, 
                 bias=True, dtype=torch.float32):
        super().__init__()
        
        
        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, (out_features, in_features), dtype=torch.int8
            )
        )
        
        self.register_buffer("scales", 
                             torch.randn((out_features), dtype=dtype))
        
        if bias:
            self.register_buffer("bias", 
                                 torch.randn((1, out_features), 
                                             dtype=dtype))
        
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights
                        /scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales
    
    def forward(self, input):
        return w8_a16_forward(self.int8_weights, 
                              input, self.scales, self.bias)      
        

if __name__ == "__main__":
    layer = W8A16LinearLayer(4, 8, bias=True)
    weights = torch.randn((8, 4), dtype=torch.float32)
    print("Weights before:\n" , weights)
    layer.quantize(weights)
    print("Weights After:\n" , layer.int8_weights)
    
    input = torch.randn((2, 4), dtype=torch.float32)
    output = layer(input)
    print("Output:\n", output)
    print("Output shape:\n", output.shape)
    print("Output dtype:\n", output.dtype)
