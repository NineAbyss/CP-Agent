# Per-Component Token and Cost Breakdown

This table attributes CP-Agent's API token usage and cost across its four main
components on LCB-Pro. Results use the DeepSeek-V3.2-Chat backbone and average
over 167 problems.

HP and SV are non-LLM tools for code execution and validation; their token
counts measure the associated LLM reasoning overhead, including accumulated
multi-turn conversation context. TA and Experience are separate LLM calls with
self-contained prompts. Costs are computed using DeepSeek API pricing.

| Component | Avg prompt tokens | Avg completion tokens | Cost / problem | Percent of total cost |
|---|---:|---:|---:|---:|
| HP (Hypothesis Prober) | 392,076 | 17,898 | `$0.025` | 69.7% |
| SV (Solution Validator) | 115,532 | 2,826 | `$0.006` | 17.6% |
| Exp (Experience) | 20,477 | 706 | `$0.001` | 3.4% |
| TA (Test Augmentation) | 9,078 | 471 | `$0.001` | 1.7% |
| Total tracked | 537,163 | 21,901 | `$0.033` | 92.4% |

The tracked components account for 92.4% of total API cost. The remaining
approximately 7.6% comes from problem reading, system prompts, and submission
formatting. HP dominates cost because it is used repeatedly inside the
multi-turn refinement loop, while TA and Experience together account for only
5.1% of total cost.

