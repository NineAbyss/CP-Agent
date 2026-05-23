# Frontier-Backbone Transfer

This experiment applies CP-Agent to the stronger DeepSeek-V3.2 Reasoner
backbone. The goal is to test whether CP-Agent only helps weaker chat backbones
or remains useful when the base model already has stronger reasoning ability.

| Model | ICPC-Eval Refine@5 | #T | Easy | Medium | Hard | LCB-Pro Avg |
|---|---:|---:|---:|---:|---:|---:|
| Reasoner base | 33.9 | 1.6 | 84.5 | 24.6 | 7.7 | 58.1 |
| CP-Agent + Reasoner | 43.2 | 1.8 | 91.8 | 31.6 | 7.7 | 64.7 |
| Delta | +9.3 | - | +7.3 | +7.0 | +0.0 | +6.6 |

CP-Agent improves the Reasoner backbone by `+9.3` Refine@5 on ICPC-Eval and
`+6.6` average Pass@1 on LCB-Pro. Gains concentrate on Easy and Medium
problems; Hard remains unchanged, consistent with the paper's conclusion that
hard problems require stronger high-level algorithmic discovery beyond feedback
utilization alone.

