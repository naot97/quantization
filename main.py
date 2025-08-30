import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from quantizer import (
    W8A16LinearLayer,
    replace_linear_with_target_and_quantize
)        
            
class DummyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(1, 1)
    # Try with bias
    self.linear_1 = nn.Linear(1, 1)
    # Try without bias
    self.linear_2 = nn.Linear(1, 1, bias=False)
    # Lm prediction head
    self.lm_head = nn.Linear(1, 1, bias=False)
    

if __name__ == "__main__":

    model_1 = DummyModel()
    model_2 = DummyModel()
    print("Loading a dummy model...\n")
    replace_linear_with_target_and_quantize(model_1, W8A16LinearLayer, ["lm_head"])
    print("model_1: ", model_1)
    replace_linear_with_target_and_quantize(model_2, W8A16LinearLayer, [])
    print("model_2: ", model_2)
    
    print("\n\nLoading a real model from Huggingface...\n")
    model_id = "Salesforce/codegen-350M-mono"
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                    torch_dtype=torch.bfloat16, 
                                             low_cpu_mem_usage=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    previous_memory_footprint = model.get_memory_footprint()
    print("Memory Footprint before quantization: ", 
          previous_memory_footprint / (1024**3), " GB")
    print("Output before quantization: ",
          pipe("def hello_world():", max_new_tokens=20, 
               do_sample=False)[0]["generated_text"])
    replace_linear_with_target_and_quantize(model, 
                                        W8A16LinearLayer, ["lm_head"])
    current_memory_footprint = model.get_memory_footprint()
    print("Memory Footprint after quantization: ", 
          current_memory_footprint / (1024**3), " GB")
    print("Output after quantization: ",
          pipe("def hello_world():", max_new_tokens=20, 
               do_sample=False)[0]["generated_text"]) 
    print("Reduction in memory footprint: ",
          (previous_memory_footprint - current_memory_footprint) / (1024**3), " GB")
    
    