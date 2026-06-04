# Fine-Tuning Empirical Analysis Report
**Target Language:** Newari (नेपाल भाषा)  
**Base Architecture:** Google Gemma-3-270M  
**Methodology:** Parameter-Efficient QLoRA Tracking via Custom Interceptors  

## 1. Executive Experimental Summary
This report aggregates behavioral and telemetry properties during the custom alignment phase of an edge-optimized language model for Newari token autocompletion.

## 2. Model & Optimization Parameters
| Parameter | Value Spec |
| :--- | :--- |
| Base Foundation Model | `google/gemma-3-270m` |
| Total Base Parameters | 271,895,808 |
| Trainable Parameters (LoRA) | 3,796,992 |
| Parameter Representation % | 1.3965% |
| Precision Matrix | BFloat16 Native |
| Optimization Strategy | Causal LM with Sequence-Wide Teacher Forcing |

## 3. Training & Convergence Analytics
| Quantitative Metric | Value Outcome |
| :--- | :--- |
| Initial Cross-Entropy Loss | 7.3691 |
| Terminal Cross-Entropy Loss | 2.3963 |
| Absolute Loss Delta ($\Delta$) | 4.9728 |
| Total Steps Executed | 1790 |
| Complete Run Runtime | 12.68 Minutes |

## 4. On-Device Hardware & Sustainability Compute Metrics
| Telemetry Component | Measured Benchmarks |
| :--- | :--- |
| Peak VRAM Required | 10.344 GB |
| Mean Operational GPU Power | 40.70 Watts |
| Total Estimated Energy Footprint | 8.605 Wh |

## 5. Artifact Output Manifest
The following generated figure structures are preserved in high-resolution format (300 DPI) for document composition:
- `fig_loss_convergence.png`: Line plotting mapping cross-entropy step optimization against the learning rate decay curve.
- `fig_hardware_footprint.png`: Double y-axis telemetry showcasing execution runtime tracking of physical VRAM load profiles vs dynamic thermal power draws.

---
*Report auto-compiled for academic peer-review layout.*
