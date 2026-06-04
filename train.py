import os
import time
import json
import torch
import pynvml
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Configuration Constants
MODEL_ID = "google/gemma-3-270m"
TOKENIZER_DIR = "./tokenizer_dir"
DATASET_PATH = "newari_dataset.jsonl"
OUTPUT_DIR = "./gemma3-270m-newari-pipeline"
METRICS_JSON_PATH = "training_metrics.json"
SUMMARY_JSON_PATH = "final_summary.json"
REPORT_MD_PATH = "research_report.md"

class HardwareTelemetryCallback(TrainerCallback):
    """
    Custom Hugging Face Trainer callback to record hardware operational telemetry
    and step-wise loss metrics concurrently.
    """
    def __init__(self):
        super().__init__()
        self.metrics_log = []
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                
                telemetry_entry = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "timestamp": time.time(),
                    "vram_used_gb": mem_info.used / (1024**3),
                    "gpu_power_watts": power_draw,
                    **logs
                }
                self.metrics_log.append(telemetry_entry)
            except pynvml.NVMLError as nv_err:
                print(f"[Warning] Failed to query NVML metrics: {nv_err}")
                telemetry_entry = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "timestamp": time.time(),
                    "vram_used_gb": 0.0,
                    "gpu_power_watts": 0.0,
                    **logs
                }
                self.metrics_log.append(telemetry_entry)

    def on_train_end(self, args, state, control, **kwargs):
        with open(METRICS_JSON_PATH, "w") as f:
            json.dump(self.metrics_log, f, indent=4)
        pynvml.nvmlShutdown()


def verify_environment():
    """Validates CUDA hardware availability and essential input file architecture."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA execution environment not detected. Aborting pipeline.")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Target training file '{DATASET_PATH}' missing from directory.")
    if not os.path.exists(TOKENIZER_DIR):
        raise FileNotFoundError(f"Specified tokenizer directory '{TOKENIZER_DIR}' not found.")


def build_plots_and_report(total_time_sec, param_stats):
    """
    Processes saved telemetry JSON payloads to output publication-grade
    visualization assets and academic Markdown documents.
    """
    if not os.path.exists(METRICS_JSON_PATH):
        print("[Error] Telemetry artifacts missing. Report generation aborted.")
        return

    df = pd.DataFrame(json.load(open(METRICS_JSON_PATH, "r")))
    df_loss = df[df['loss'].notna()].copy()

    if df_loss.empty:
        print("[Error] Insufficient step data logs to interpolate plot points.")
        return

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2

    # Figure 1: Convergence and Learning Rate Decay Schedule
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel('Training Steps', fontweight='bold')
    ax1.set_ylabel('Cross-Entropy Loss', color='#1f77b4', fontweight='bold')
    ax1.plot(df_loss['step'], df_loss['loss'], color='#1f77b4', linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if 'learning_rate' in df_loss.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', color='#d62728', fontweight='bold')
        ax2.plot(df_loss['step'], df_loss['learning_rate'], color='#d62728', linestyle=':', linewidth=1.5, label='LR')
        ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.title('Language Model Convergence & LR Schedule', fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig('fig_loss_convergence.png', dpi=300)
    plt.close()

    # Figure 2: Hardware Telemetry Metrics (VRAM Profile vs Power consumption)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel('Training Steps', fontweight='bold')
    ax1.set_ylabel('VRAM Allocation (GB)', color='#2ca02c', fontweight='bold')
    ax1.plot(df_loss['step'], df_loss['vram_used_gb'], color='#2ca02c', linewidth=2, label='VRAM')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if 'gpu_power_watts' in df_loss.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('GPU Power Draw (Watts)', color='#ff7f0e', fontweight='bold')
        ax2.plot(df_loss['step'], df_loss['gpu_power_watts'], color='#ff7f0e', alpha=0.7, linewidth=1.5, label='Power')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    plt.title('Hardware Operational Footprint on Edge Fine-Tuning', fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig('fig_hardware_footprint.png', dpi=300)
    plt.close()

    avg_power = df_loss['gpu_power_watts'].mean() if 'gpu_power_watts' in df_loss.columns else 0.0
    peak_vram = df_loss['vram_used_gb'].max()
    initial_loss = df_loss['loss'].iloc[0]
    final_loss = df_loss['loss'].iloc[-1]
    total_energy_wh = avg_power * (total_time_sec / 3600.0)

    report_md = f"""# Fine-Tuning Empirical Analysis Report
**Target Language:** Newari (नेपाल भाषा)  
**Base Architecture:** Google Gemma-3-270M  
**Methodology:** Parameter-Efficient QLoRA Tracking via Custom Interceptors  

## 1. Executive Experimental Summary
This report aggregates behavioral and telemetry properties during the custom alignment phase of an edge-optimized language model for Newari token autocompletion.

## 2. Model & Optimization Parameters
| Parameter | Value Spec |
| :--- | :--- |
| Base Foundation Model | `{MODEL_ID}` |
| Total Base Parameters | {param_stats['total_params']:,} |
| Trainable Parameters (LoRA) | {param_stats['trainable_params']:,} |
| Parameter Representation % | {param_stats['param_percentage']:.4f}% |
| Precision Matrix | BFloat16 Native |
| Optimization Strategy | Causal LM with Sequence-Wide Teacher Forcing |

## 3. Training & Convergence Analytics
| Quantitative Metric | Value Outcome |
| :--- | :--- |
| Initial Cross-Entropy Loss | {initial_loss:.4f} |
| Terminal Cross-Entropy Loss | {final_loss:.4f} |
| Absolute Loss Delta ($\Delta$) | {initial_loss - final_loss:.4f} |
| Total Steps Executed | {int(df_loss['step'].max())} |
| Complete Run Runtime | {total_time_sec / 60.0:.2f} Minutes |

## 4. On-Device Hardware & Sustainability Compute Metrics
| Telemetry Component | Measured Benchmarks |
| :--- | :--- |
| Peak VRAM Required | {peak_vram:.3f} GB |
| Mean Operational GPU Power | {avg_power:.2f} Watts |
| Total Estimated Energy Footprint | {total_energy_wh:.3f} Wh |

## 5. Artifact Output Manifest
The following generated figure structures are preserved in high-resolution format (300 DPI) for document composition:
- `fig_loss_convergence.png`: Line plotting mapping cross-entropy step optimization against the learning rate decay curve.
- `fig_hardware_footprint.png`: Double y-axis telemetry showcasing execution runtime tracking of physical VRAM load profiles vs dynamic thermal power draws.

---
*Report auto-compiled for academic peer-review layout.*
"""
    with open(REPORT_MD_PATH, "w") as f:
        f.write(report_md)


def main():
    verify_environment()

    print(f"[Pipeline] Loading local tokenizer from directory: {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR,trust_remote_code="True")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[Pipeline] Initializing Base Model ({MODEL_ID}) on CUDA...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
   
    
    _tmp_model = get_peft_model(model, peft_config)
    trainable_p, total_p = _tmp_model.get_nb_trainable_parameters()
    del _tmp_model
    param_stats = {
        "trainable_params": trainable_p,
        "total_params": total_p,
        "param_percentage": (trainable_p / total_p) * 100
    }

    print("[Pipeline] Preparing Newari Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Native SFTConfig encapsulates dataset properties and transformer parameters
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=5,
        optim="adamw_torch",
        bf16=True,
        dataloader_pin_memory=True,
        save_strategy="no",
        report_to="none",
    )

    telemetry_callback = HardwareTelemetryCallback()

    # Initialization interface simplified to fulfill SFTConfig structural changes
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        
        args=sft_config,
        peft_config=peft_config,
        callbacks=[telemetry_callback]
    )

    print("[Pipeline] Training initiated...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    total_time_sec = end_time - start_time
    print("[Pipeline] Training loop complete.")

    summary_metrics = {
        "total_training_time_sec": total_time_sec,
        "train_loss": train_result.training_loss,
        **param_stats
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary_metrics, f, indent=4)

    print("[Pipeline] Processing telemetry arrays and building report artifacts...")
    build_plots_and_report(total_time_sec, param_stats)
    
    print("[Pipeline] Saving model adapter weights and custom tokenizer...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[Pipeline] Process complete. Artifacts saved.")


if __name__ == "__main__":
    main()
