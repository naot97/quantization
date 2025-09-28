#!/usr/bin/env python3
"""
Evaluation script for Llama 3.1 3B with different quantization methods.
Based on TensorRT-Model-Optimizer quantization method recommendations.
"""

import sys
import os
import torch
import time
import json
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add TensorRT-Model-Optimizer to Python path
sys.path.insert(0, '/home/naot/myspace/quantization/TensorRT-Model-Optimizer')

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
    get_max_batch_size,
)


class QuantizationEvaluator:
    def __init__(self, model_name="meta-llama/Llama-3.1-3B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Test prompts for evaluation
        self.test_prompts = [
            "The future of artificial intelligence is",
            "Climate change is a global challenge that requires",
            "In the field of quantum computing, researchers have discovered",
            "The benefits of renewable energy include",
            "Machine learning algorithms can be used to"
        ]
    
    def load_model_with_quantization(self, quant_method):
        """Load model with specified quantization method using TensorRT-Model-Optimizer."""
        print(f"Loading model with {quant_method} quantization...")
        
        # Load base model first
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if quant_method == "baseline":
            return model
        
        # Get quantization configuration
        quant_cfg = self.get_quantization_config(quant_method)
        
        if quant_cfg is None:
            print(f"Quantization method {quant_method} not supported, returning baseline model")
            return model
        
        # Setup calibration data
        calib_dataloader = get_dataset_dataloader(
            dataset_name=["cnn_dailymail"],
            tokenizer=self.tokenizer,
            batch_size=4,
            num_samples=[128],
            device=model.device,
        )
        
        # Create forward loop for calibration
        calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
        
        # Apply quantization
        print(f"Applying {quant_method} quantization...")
        start_time = time.time()
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
        end_time = time.time()
        print(f"Quantization completed in {end_time - start_time:.2f}s")
        
        return model
    
    def get_quantization_config(self, quant_method):
        """Get quantization configuration for the specified method."""
        configs = {
            "fp8": mtq.FP8_DEFAULT_CFG,
            "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
            "int4_awq": mtq.INT4_AWQ_CFG,
            "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        }
        
        if quant_method == "int8_smoothquant":
            return configs["int8_sq"]
        elif quant_method == "int4_fp8_awq":
            return configs["w4a8_awq"]
        
        return configs.get(quant_method)
    
    def measure_memory_usage(self):
        """Measure current GPU memory usage."""
        gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        return gpu_memory
    
    def evaluate_batch_performance(self, model, batch_sizes=[1, 4, 16, 32]):
        """Evaluate model performance across different batch sizes."""
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Prepare batch
            prompts = self.test_prompts[:batch_size] if batch_size <= len(self.test_prompts) else self.test_prompts * (batch_size // len(self.test_prompts) + 1)
            prompts = prompts[:batch_size]
            
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Warmup
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=50, do_sample=False)
            
            # Measure performance
            start_time = time.time()
            gpu_mem_before = self.measure_memory_usage()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            end_time = time.time()
            gpu_mem_after = self.measure_memory_usage()
            
            # Calculate metrics
            inference_time = end_time - start_time
            tokens_generated = batch_size * 100  # 100 new tokens per sequence
            tokens_per_second = tokens_generated / inference_time
            
            results[batch_size] = {
                "inference_time": inference_time,
                "tokens_per_second": tokens_per_second,
                "gpu_memory_used": gpu_mem_after - gpu_mem_before,
            }
            
            # Clean up
            del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def evaluate_quantization_method(self, quant_method):
        """Evaluate a specific quantization method."""
        print(f"\n{'='*60}")
        print(f"Evaluating {quant_method.upper()} quantization")
        print(f"{'='*60}")
        
        # Load model
        start_load = time.time()
        model = self.load_model_with_quantization(quant_method)
        load_time = time.time() - start_load
        
        # Get model size
        model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        
        print(f"Model loaded in {load_time:.2f}s")
        print(f"Model size: {model_size_gb:.2f} GB")
        
        # Evaluate performance
        results = self.evaluate_batch_performance(model)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "load_time": load_time,
            "model_size_gb": model_size_gb,
            "batch_results": results
        }
    
    def run_full_evaluation(self):
        """Run evaluation for all quantization methods."""
        methods = ["baseline", "fp8", "int8_smoothquant", "int4_awq", "int4_fp8_awq"]
        all_results = {}
        
        print(f"Starting evaluation of Llama 3.1 3B quantization methods")
        print(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        for method in methods:
            try:
                all_results[method] = self.evaluate_quantization_method(method)
            except Exception as e:
                print(f"Error evaluating {method}: {e}")
                all_results[method] = {"error": str(e)}
        
        # Print summary
        self.print_summary(all_results)
        return all_results
    
    def print_summary(self, results):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Method':<20} {'Size (GB)':<12} {'Load Time':<12} {'Tokens/s (B=1)':<15} {'Tokens/s (B=16)':<15}")
        print("-" * 80)
        
        for method, data in results.items():
            if "error" in data:
                print(f"{method:<20} ERROR: {data['error']}")
                continue
                
            size = f"{data['model_size_gb']:.2f}"
            load_time = f"{data['load_time']:.2f}s"
            
            # Get tokens/s for batch size 1 and 16
            batch_1_tps = data['batch_results'].get(1, {}).get('tokens_per_second', 0)
            batch_16_tps = data['batch_results'].get(16, {}).get('tokens_per_second', 0)
            
            print(f"{method:<20} {size:<12} {load_time:<12} {batch_1_tps:<15.1f} {batch_16_tps:<15.1f}")
        
        print("\nRecommendations based on results:")
        print("- Small batch (≤4): Use INT4 AWQ or INT4-FP8 AWQ")
        print("- Large batch (≥16): Use FP8 or INT8 SmoothQuant")
        print("- Best accuracy: FP8 (if GPU supports it)")
        print("- Smallest size: INT4 methods (25% of original)")


def main():
    evaluator = QuantizationEvaluator()
    results = evaluator.run_full_evaluation()
    
    # Save results
    with open("llama31_3b_quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to llama31_3b_quantization_results.json")


if __name__ == "__main__":
    main()