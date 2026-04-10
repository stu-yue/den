import torch
import time
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer

# --- 核心配置 ---
DEVICE = "cuda"
DTYPE = torch.float16 

# 审稿人关注的压力测试矩阵
SEQ_LENGTHS = [128, 512]
BATCH_SIZES = [8, 32, 128, 512, 1024] 
NUM_WARMUP = 10
NUM_STEPS = 50

def get_gpu_memory():
    """返回当前显存占用 (MB)"""
    return torch.cuda.memory_allocated(DEVICE) / 1024**2

def benchmark_model(model_path):
    print(f"\n🚀 开始性能评估: {model_path}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. 静态显存加载 (Static Memory Footprint)
    # 对应审稿人要求的 "all experts must be loaded"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=DTYPE
    ).to(DEVICE).eval()
    
    static_mem = torch.cuda.memory_allocated(DEVICE) / 1024**2
    print(f"✅ 静态显存占用: {static_mem:.2f} MB")

    all_results = []

    # 2. 性能压力测试
    for slen in SEQ_LENGTHS:
        print(f"\n测试序列长度 (Seq_Len): {slen}")
        print(f"{'BS':<8} | {'Latency (ms)':<15} | {'Throughput (s/s)':<18} | {'Peak Mem (MB)':<12}")
        print("-" * 60)
        
        for bs in BATCH_SIZES:
            try:
                # 构造符合长度要求的 Dummy 数据
                input_ids = torch.randint(0, 1000, (bs, slen)).to(DEVICE)
                attention_mask = torch.ones((bs, slen)).to(DEVICE)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

                # 预热 (Warmup)
                for _ in range(NUM_WARMUP):
                    with torch.no_grad():
                        _ = model(**inputs)
                
                # 启动 CUDA Event 计时
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
                
                avg_latency = np.mean(latencies)
                throughput = (bs * 1000) / avg_latency # 每秒处理句子数
                peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

                all_results.append({
                    "Seq_Len": slen,
                    "Batch_Size": bs,
                    "Latency_ms": round(avg_latency, 2),
                    "Throughput_fps": round(throughput, 2),
                    "Peak_Mem_MB": round(peak_mem, 2)
                })
                
                print(f"{bs:<8} | {avg_latency:<15.2f} | {throughput:<18.2f} | {peak_mem:<12.2f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{bs:<8} | {'❌ OOM':<15} | {'-':<18} | {'-':<12}")
                    torch.cuda.empty_cache()
                    break # 跳过该 SeqLen 下更大的 BS
                else:
                    raise e

    return all_results

if __name__ == "__main__":
    qwen3_06B_path = "/root/models/Qwen3-Embedding-0.6B"
    qwen3_4B_path = "/root/models/Qwen3-Embedding-4B"
    results = benchmark_model(qwen3_06B_path)
    results = benchmark_model(qwen3_4B_path)
    
    
    df = pd.DataFrame(results)
    print("\n--- 最终性能汇总表 ---")
    print(df.to_string(index=False))