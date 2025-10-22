# mkdir -p mxfp4-res
# # task list gsm8k, aime25, gpqa, mmlu

# task_list=gsm8k,aime25,gpqa,mmlu


# OPENAI_API_KEY=None  python -m gpt_oss.evals --model /data6/yiliu4/unsloth-gpt-oss-120b-BF16-ar-MXFP4/  --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low --output_dir mxfp4-res

#!/bin/bash
set -e  # Exit immediately if any command fails

# Create results directory
mkdir -p mxfp4-res

# Define task list
task_list="gsm8k,aime25,gpqa,mmlu"
task_list="mmlu"
MODEL_PATH="/data6/yiliu4/unsloth-gpt-oss-120b-BF16-ar-MXFP4/"
MODEL_PATH="/storage/yiliu7/unsloth/gpt-oss-120b-BF16-ar-MXFP4"
API_ENDPOINT="http://localhost:8099/v1"

# Log start time
echo "Starting evaluation at $(date)"
echo "Testing model: $MODEL_PATH"
echo "=============================================="

# Process each task in the task list
IFS=',' read -ra TASKS <<< "$task_list"
for task in "${TASKS[@]}"; do
    echo "Running evaluation for task: $task"
    echo "--------------------------------------------"
    
    # Run evaluation for current task
    OPENAI_API_KEY=None python -m gpt_oss.evals \
        --model "$MODEL_PATH" \
        --eval "$task" \
        --base-url "$API_ENDPOINT" \
        --reasoning-effort low \
        --n-threads 8 \
        --output_dir "./mxfp4-res" \
        2>&1 | tee "mxfp4-res/${task}_log.txt"
        
    echo "Completed $task at $(date)"
    echo "=============================================="
done

echo "All evaluations complete!"
echo "Results saved in mxfp4-res/"