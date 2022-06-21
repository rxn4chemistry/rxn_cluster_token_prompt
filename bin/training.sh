conda activate rxn-cluster-token-prompt

# DATA LOCATION
TRAIN_SRC="" # the txt file containing the tokenized products for training
TRAIN_TGT="" # the txt file containing the tokenized precursors for training
VALID_SRC="" # the txt file containing the tokenized products for validation
VALID_TGT="" # the txt file containing the tokenized precursors for validation

PREPROCESSDIR="" # were to store the preprocessed files and the vocab

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
OUTPUTDIR="" # where to store the trained model

# TRAINING
onmt_train -data ${PREPROCESSDIR}/preprocessed_rxn_cluster_token_prompt \
           -save_model  ${OUTPUTDIR}/model \
           -seed 42 -gpu_ranks 0 -save_checkpoint_steps 5000 -keep_checkpoint 1 \
           -train_steps 130000 -param_init 0  -param_init_glorot -max_generator_batches 32 \
           -batch_size 6144 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
           -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
           -learning_rate 2 -label_smoothing 0.0 -report_every 1000  -valid_batch_size 8 \
           -layers 4 -rnn_size  384 -word_vec_size 384 -encoder_type transformer -decoder_type transformer \
           -dropout 0.1 -position_encoding -share_embeddings \
           -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
           -heads 8 -transformer_ff 2048