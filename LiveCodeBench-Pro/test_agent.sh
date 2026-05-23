#!/bin/bash
cd LiveCodeBench-Pro


CONFIG_PATH="agentflow/configs/config.yaml"


JUDGE_INSTANCE_ID="${JUDGE_INSTANCE_ID:-new_testcase}"

CONFIG_NAME=$(basename "${CONFIG_PATH}" .yaml)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

export PHOENIX_PROJECT_NAME="LCB-Pro_${CONFIG_NAME}_${TIMESTAMP}"

export LCB_PRO_MODE=1


python run_agentflow_benchmark_parallel.py \
    --config "${CONFIG_PATH}" \
    --split quater_2025_4_6 \
    --difficulty easy medium hard \
    --parallel 8 \
    --worker 8 \
    --judge-instance-id "${JUDGE_INSTANCE_ID}"  

echo ""
