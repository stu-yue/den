import torch
import time
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 让脚本可以直接 import dense2moe/den2moee 下的自定义结构
SCRIPT_DIR = Path(__file__).resolve().parent
MOE2_SRC = SCRIPT_DIR / "dense2moe"
if not MOE2_SRC.exists():
    raise FileNotFoundError(
        f"未找到目录: {MOE2_SRC}"
    )

if str(MOE2_SRC) not in sys.path:
    sys.path.insert(0, str(MOE2_SRC))

# 触发 den2moee 的 AutoConfig / AutoModel / AutoModelForCausalLM 注册
from den2moee.configuration_den2moee import Den2MoEEConfig  # noqa: F401
from den2moee.modeling_den2moee import Den2MoEEForCausalLM  # noqa: F401

# --- 核心配置 ---
DEVICE = "cuda"
DTYPE = torch.float16 
SEQ_LENGTHS = [128, 512]
BATCH_SIZES = [8, 32, 128, 512, 1024] 
NUM_WARMUP = 10
NUM_STEPS = 50
LOG_FILE = "outputs/sys_metrics.log"

def get_gpu_memory():
    return torch.cuda.memory_allocated(DEVICE) / 1024**2

def benchmark_model(model_path):
    # 提取模型名称用于日志标注
    model_name = os.path.basename(model_path.rstrip("/"))
    print(f"\n🚀 开始性能评估: {model_name}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. 静态显存加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # den2moee 使用 CausalLM 接口加载，其他保持原 AutoModel 行为
        if getattr(config, "model_type", None) == "den2moee":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=DTYPE,
            ).to(DEVICE).eval()
        else:
            model = AutoModel.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=DTYPE,
            ).to(DEVICE).eval()
    except Exception as e:
        print(f"❌ 加载模型 {model_name} 失败: {e}")
        return []
    
    static_mem = torch.cuda.memory_allocated(DEVICE) / 1024**2
    print(f"✅ 静态显存占用: {static_mem:.2f} MB")

    model_results = []

    # 2. 性能压力测试
    for slen in SEQ_LENGTHS:
        print(f"\n测试序列长度 (Seq_Len): {slen}")
        for bs in BATCH_SIZES:
            try:
                input_ids = torch.randint(0, 1000, (bs, slen)).to(DEVICE)
                attention_mask = torch.ones((bs, slen)).to(DEVICE)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

                for _ in range(NUM_WARMUP):
                    with torch.no_grad():
                        _ = model(**inputs)
                
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
                throughput = (bs * 1000) / avg_latency
                peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

                res = {
                    "Model": model_name,
                    "Seq_Len": slen,
                    "Batch_Size": bs,
                    "Latency_ms": round(avg_latency, 2),
                    "Throughput_fps": round(throughput, 2),
                    "Peak_Mem_MB": round(peak_mem, 2),
                    "Static_Mem_MB": round(static_mem, 2)
                }
                model_results.append(res)
                print(f"BS={bs:<5} | Latency={avg_latency:>8.2f}ms | Throughput (samples/s)={throughput:>10.2f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"BS={bs:<5} | ❌ OOM")
                    torch.cuda.empty_cache()
                    break
                else: raise e
    
    # 释放显存以供下一个模型使用
    del model
    torch.cuda.empty_cache()
    return model_results

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    model_paths = [
        # "/root/models/Qwen3-Embedding-0.6B",
        # "/root/models/Qwen3-Embedding-4B",
        "/root/code/output/Den2MoEE-Embedding-0.6B-svd-init-0.0-rank-0.4",
    ]
    
    all_metrics = []
    for path in model_paths:
        if os.path.exists(path):
            all_metrics.extend(benchmark_model(path))
        else:
            print(f"⚠️ 路径不存在: {path}")
    
    # 汇总结果并写入日志
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        summary_str = f"\n{'='*20} 性能测试汇总 ({time.strftime('%Y-%m-%d %H:%M:%S')}) {'='*20}\n"
        summary_str += df.to_string(index=False)
        summary_str += "\n\n"
        
        # 写入文件
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(summary_str)
        
        print(f"\n✅ 所有结果已汇总至: {LOG_FILE}")
        print(df.to_string(index=False))