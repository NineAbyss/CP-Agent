# More Analysis

This directory contains the additional empirical analyses referenced by the
paper's "More Analysis" section. These materials complement the calibrated
certificate and do not change the declared controller manifest or the reported
certificate value.

## Contents

1. [Section 2 proofs](proof.md)
   - Full proofs for the calibrated risk-controlled theory of feedback control,
     covering the calibration estimators after Definition 2.4, Theorem 2.5,
     Theorem 2.6, Propositions 2.7–2.9, Corollary 2.10, and the validity of the
     mechanism factorization in Eq. (9).
   - Includes the probability-space setup, stopped-process conventions,
     admission-gate semantics, held-out exchangeability, and Clopper–Pearson
     UCB/LCB tooling used throughout the proofs.

2. [Test-Augmentation diagnostics](test_augmentation_diagnostics.md)
   - Candidate-level verifier quality on 117 public-test-passing candidates.
   - Per-input sample-size diversity analysis over TA-generated tests.

3. [Dual-Granularity Verification robustness](dgv_counterfactual_robustness.md)
   - End-to-end counterfactual study over the 19 verifier-level false-positive
     TA cases.
   - Problem 2098B case study showing recovery from a spurious TA alarm.

4. [Frontier-backbone transfer](frontier_backbone_transfer.md)
   - CP-Agent applied to DeepSeek-V3.2 Reasoner, improving ICPC-Eval Refine@5
     from `33.9` to `43.2` and LCB-Pro average Pass@1 from `58.1` to `64.7`.

5. [Efficiency and BCC derivation](efficiency_bcc_derivation.md)
   - Input-oriented VRS/BCC DEA formulation behind the multi-backbone
     Pass@1-cost frontier.
   - DeepSeek per-method cost/performance scores and closed-form theta values.

6. [Per-component token and cost breakdown](component_token_cost.md)
   - Average API token and cost attribution across HP, SV, Experience, and TA.

7. [Tool use vs. problem difficulty](tool_use_difficulty.md)
   - Average HP/SV/TA calls per problem on LCB-Pro by Easy/Medium/Hard band
     (DeepSeek-V3.2-Chat backbone), with adaptive-scaling and Hard-drop
     analysis.

8. [Experience-driven efficiency](experience_efficiency.md)
   - LCB-Pro refinement-step and per-problem cost comparison with vs. without
     the frozen Experience snapshot, with mechanism interpretation tied to the
     admission gate.


