


QUANT_CONFIG_FILE="./quant_configs/inc_unit_scale.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="quant.pile.512.${timestamp}.log"

# remove ./scripts/nc_workspace_measure_kvache if needed
if [ -e ./scripts/nc_workspace_measure_kvache ]; then
    echo "The directory ./scripts/nc_workspace_measure_kvache already exists, removing it..."
    rm -rf ./scripts/nc_workspace_measure_kvache
fi


echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"



echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"

# QUANT_CONFIG=${QUANT_CONFIG_FILE} \
VLLM_DISABLE_MARK_SCALES_AS_CONST=1 \
VLLM_LOGGING_LEVEL=DEBUG \
PT_HPU_LAZY_MODE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=0 \
VLLM_SKIP_WARMUP=true  python basic.py