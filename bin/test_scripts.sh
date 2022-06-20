conda activate rxn-cluster-token-prompt

# DATA LOCATION
TRAIN_SRC="${PWD}/data/test_data/product-train.txt"
TRAIN_TGT="${PWD}/data/test_data/precursors-train.txt"
VALID_SRC="${PWD}/data/test_data/product-valid.txt"
VALID_TGT="${PWD}/data/test_data/precursors-valid.txt"

PREPROCESSDIR="${PWD}/data/test_data/preprocess" # were to store the preprocessed files and the vocab
mkdir ${PREPROCESSDIR}

# ONMT PREPROCESS
onmt_preprocess -train_src ${TRAIN_SRC} \
		-train_tgt ${TRAIN_TGT} \
		-valid_src ${VALID_SRC} \
		-valid_tgt ${VALID_TGT} \
		-save_data ${PREPROCESSDIR}/preprocessed_rxn_cluster_token_prompt \
		-src_seq_length 3000 \
		-tgt_seq_length 3000 \
		-src_vocab_size 3000 \
		-tgt_vocab_size 3000 \
		-share_vocab \
		-overwrite

# RETRO TRAINING PARAMETERS
OUTPUTDIR="${PWD}/data/test_data/training" # where to store the trained model

# TRAINING
onmt_train -data ${PREPROCESSDIR}/preprocessed_rxn_cluster_token_prompt \
           -save_model  ${OUTPUTDIR}/model \
           -seed 42 -gpu_ranks 0 -save_checkpoint_steps 10 -keep_checkpoint 1 \
           -train_steps 50 -param_init 0  -param_init_glorot -max_generator_batches 32 \
           -batch_size 6144 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
           -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
           -learning_rate 2 -label_smoothing 0.0 -report_every 1000  -valid_batch_size 8 \
           -layers 4 -rnn_size  384 -word_vec_size 384 -encoder_type transformer -decoder_type transformer \
           -dropout 0.1 -position_encoding -share_embeddings \
           -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
           -heads 8 -transformer_ff 2048


# DATA LOCATION
PRODUCTS_FILE="${PWD}/data/test_data/product-valid.txt"
PRECURSORS_FILE="${PWD}/data/test_data/precursors-valid.txt"
OUTPUT_DIR="${PWD}/data/test_data/results"
RETRO_MODEL_FILE="${PWD}/models/10clusters/10clusters.pt"
FORWARD_MODEL_FILE="${PWD}/models/forwardUSPTO/forwardUSPTO.pt"
CLASSIFICATION_MODEL_FILE="${PWD}/models/classificationUSPTO/classificationUSPTO.pt"
N_BEST=2
BEAM_SIZE=10
TOKENS=10

prepare-retro-metrics --precursors_file ${PRECURSORS_FILE} \
                      --products_file ${PRODUCTS_FILE} \
                      --output_dir ${OUTPUT_DIR} \
                      --retro_model ${RETRO_MODEL_FILE} \
                      --forward_model ${FORWARD_MODEL_FILE} \
                      --classification_model ${CLASSIFICATION_MODEL_FILE} \
                      --n_best ${N_BEST} --beam_size ${BEAM_SIZE} --gpu \
                      --class_tokens ${TOKENS}

RESULTS_DIR="${PWD}/data/test_data/results" # Directory where the inference output files are stored
GROUND_TRUTH_FILE="${RESULTS_DIR}/gt_products.txt"
PREDICTIONS_FILE="${RESULTS_DIR}/predicted_precursors_canonical.txt"
CONFIDENCES_FILE="${RESULTS_DIR}/predicted_precursors.txt.tokenized_log_probs"
FWD_PREDICTIONS_FILE="${RESULTS_DIR}/predicted_products_canonical.txt"
CLASSES_PREDICTIONS_FILE="${RESULTS_DIR}/predicted_classes.txt"
N_CLASS_TOKENS=10 # Number of class tokens of the trained model

# First reorder the retrosynthesis predictions for the cluster token prompt models
# taking first all the top ones and ordering by forward likelihood and then all the top2, as so on
# The scripts saves the new files with the `.reordered` extension. The reordering needs NOT to be executed
# for the baseline model

reorder-retro-predictions-class-token --ground_truth_file ${GROUND_TRUTH_FILE} \
                                      --predictions_file ${PREDICTIONS_FILE} \
                                      --confidences_file ${CONFIDENCES_FILE} \
                                      --fwd_predictions_file ${FWD_PREDICTIONS_FILE} \
                                      --classes_predictions_file ${CLASSES_PREDICTIONS_FILE} \
                                      --n_class_tokens ${N_CLASS_TOKENS}

# Now compute the metrics: remove the `--reordered` flag to get results for the baseline model

compute-retro-metrics --results_dir ${RESULTS_DIR} \
                      --reordered