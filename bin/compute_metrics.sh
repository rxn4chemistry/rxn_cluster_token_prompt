conda activate rxn-cluster-token-prompt

RESULTS_DIR="" # Directory where the inference output files are stored
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

reorder-retro-predictions --ground_truth_file ${GROUND_TRUTH_FILE} \
                          --predictions_file ${PREDICTIONS_FILE} \
                          --confidences_file ${CONFIDENCES_FILE} \
                          --fwd_predictions_file ${FWD_PREDICTIONS_FILE} \
                          --classes_predictions_file ${CLASSES_PREDICTIONS_FILE} \
                          --n_class_tokens ${N_CLASS_TOKENS}

# Now compute the metrics: remove the `--reordered` flag to get results for the baseline model

compute-retro-metrics --results_dir ${RESULTS_DIR} \
                      --reordered
