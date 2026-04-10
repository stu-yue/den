#!/bin/bash

# Generate CUDA_VISIBLE_DEVICES as a range from 0 to NUM_GPUS-1
NUM_GPUS=8
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

# Local workspace root
HOME_DIR="/root/code/den"
WORK_DIR="${HOME_DIR}/dense2moe"

MODEL_PATH="${HOME_DIR}/../../models/Qwen3-Embedding-0.6B"
OUTPUT_DIR="${HOME_DIR}/../output"
INPUT_FILE="${HOME_DIR}/cal_combined.jsonl"
# Pin Python to a specific conda env to avoid using system site-packages.
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/py310/bin/python}"

cd "$WORK_DIR"
mkdir -p "${OUTPUT_DIR}"



while [[ $# -gt 0 ]]; do
  case $1 in
    --input-file=*)
      INPUT_FILE="${1#*=}"
      echo "New INPUT_FILE: $INPUT_FILE"
      shift
      ;;
    --output-file=*)
      OUTPUT_FILE="${1#*=}"
      echo "New OUTPUT_FILE: $OUTPUT_FILE"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

MODEL_NAME=$(basename $MODEL_PATH)

if [ -z "$OUTPUT_FILE" ]; then
  DATE_TIME=$(date +"%Y%m%d_%H%M%S")
  CALIBRATE_DATASET=$(basename $INPUT_FILE .jsonl)
  CALIBRATE_DATASET=domain
  OUTPUT_FILE=${OUTPUT_DIR}/${MODEL_NAME}_${CALIBRATE_DATASET}_${DATE_TIME}_forward.pt
  echo "OUTPUT_FILE was not set. Current OUTPUT_FILE: $OUTPUT_FILE"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "PYTHON_BIN does not exist or is not executable: $PYTHON_BIN"
  echo "Please set it explicitly, e.g. PYTHON_BIN=/path/to/miniconda3/envs/py310/bin/python"
  exit 1
fi


"$PYTHON_BIN" -m accelerate.commands.launch --num_processes $NUM_GPUS --multi_gpu forward_calibration.py \
  --model_name_or_path $MODEL_PATH \
  --input_files $INPUT_FILE \
  --output_file $OUTPUT_FILE

chmod -R 777 "$OUTPUT_DIR"
