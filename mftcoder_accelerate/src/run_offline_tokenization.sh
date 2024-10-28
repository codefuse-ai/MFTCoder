MODEL_PATH=
DATA_PATH=
DATASET_NAME=
OUTPUT_PATH=

python offline_tokenization/concat_sst_bin_tokenization.py \
--model-path ${MODEL_PATH} \
--data-path ${DATA_PATH} \
--dataset-name ${DATASET_NAME} \
--output-path ${OUTPUT_PATH} \
--parallel 16 \
--seq-length 4096 \
--sample-percent 1.0
