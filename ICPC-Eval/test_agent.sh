#!/bin/bash
cd ICPC-Eval


CONFIG_PATH="agentflow/configs/config.yaml"

CONFIG_NAME=$(basename "${CONFIG_PATH}" .yaml)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

export PHOENIX_PROJECT_NAME="${CONFIG_NAME}_${TIMESTAMP}"

LOG_DIR="ICPC-Eval/terminal_log"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"


echo "" | tee -a "${LOG_FILE}"


python eval_agent.py \
    --config "${CONFIG_PATH}" \
    --num_threads 8 \
    --dataset_name RUC-AIBOX/ICPC-Eval \
    --auto_choose  \
    2>&1 | tee -a "${LOG_FILE}"



echo "" | tee -a "${LOG_FILE}"
