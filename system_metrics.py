import torch
import time
import pandas as pd
from transformers import AutoModel, AutoTokenizer

# --- 配置 ---
MODEL_ID = "path/to/your/model" 
DEVICE = "cuda"
DTYPE = torch.float16 # 推理常用精度
SEQ_LEN = 512
BATCH_SIZES = [8, 32, 128, 512] # 测试不同压力下的表现
NUM_WARMUP = 10
NUM_STEPS = 50

def get_gpu_memory():
    # 返回当前已分配的显存 (MB)
    return torch.cuda.memory_allocated(DEVICE) / 1024**2

def benchmark_model(model_path):
    print(f"\n🚀 Testing Model: {model_path}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. 测量静态显存 (Memory Footprint)
    base_mem = get_gpu_memory()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=DTYPE
    ).to(DEVICE).eval()
    
    static_mem = get_gpu_memory() - base_mem
    print(f"✅ Memory Footprint (Static): {static_mem:.2f} MB")

    results = []

    # 2. 循环测试不同的 Batch Size
    for bs in BATCH_SIZES:
        # 构造 Dummy Input
        input_ids = torch.randint(0, 1000, (bs, SEQ_LEN)).to(DEVICE)
        attention_mask = torch.ones((bs, SEQ_LEN)).to(DEVICE)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # 预热
        for _ in range(NUM_WARMUP):
            with torch.no_grad():
                _ = model(**inputs)
        
        # 正式计时
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        latencies = []
        torch.cuda.reset_peak_memory_stats()

        for _ in range(NUM_STEPS):
            start_event.record()
            with torch.no_grad():
                _ = model(**inputs)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        
        avg_latency = sum(latencies) / len(latencies)
        throughput = (bs * 1000) / avg_latency # 每秒处理请求数
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

        results.append({
            "Batch Size": bs,
            "Latency (ms)": round(avg_latency, 2),
            "Throughput (samples/s)": round(throughput, 2),
            "Peak Memory (MB)": round(peak_mem, 2)
        })
        print(f"BS={bs}: Latency={avg_latency:.2f}ms, Throughput={throughput:.2f} samples/s")

    return results, static_mem

# --- 执行对比 ---
# 建议运行两次脚本，分别测 Qwen3 和 Den2MoEE，或者在此循环
# qwen_results, qwen_static = benchmark_model("qwen3-path")
# moe_results, moe_static = benchmark_model("den2moee-path")