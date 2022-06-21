conda activate rxn-cluster-token-prompt

# DATA LOCATION
PRODUCTS_FILE="" # The txt file containing the products to test (one per line)
PRECURSORS_FILE="" # The txt file containing the target precursors (one per line)
OUTPUT_DIR="" # The output directory
RETRO_MODEL_FILE="" # The trained retrosynthesis model
FORWARD_MODEL_FILE="" # The trained forward model
CLASSIFICATION_MODEL_FILE="" # The trained classification model
N_BEST="" # The number of predictions retained for each token . Must be <=BEAM_SIZE
BEAM_SIZE=10
TOKENS="" # The number of tokens the model was trained on

prepare-retro-metrics --precursors_file ${PRECURSORS_FILE} \
                      --products_file ${PRODUCTS_FILE} \
                      --output_dir ${OUTPUT_DIR} \
                      --retro_model ${RETRO_MODEL_FILE} \
                      --forward_model ${FORWARD_MODEL_FILE} \
                      --classification_model ${CLASSIFICATION_MODEL_FILE} \
                      --n_best ${N_BEST} --beam_size ${BEAM_SIZE} --gpu \
                      --class_tokens ${TOKENS} # comment this out if you are using a baseline retrosynthesis model to make predictions