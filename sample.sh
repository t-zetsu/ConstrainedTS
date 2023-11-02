# --- 難易度辞書の作成 ---
# python3 scripts/generate_dictionary.py \
#     --input data/datasets/sample/train.jsonl \
#     --output data/datasets/sample/train.tf.json 

# --- 入力文と正解文のワードアラインメント ---
# pip3 install transformers==3.0.2
# CUDA_VISIBLE_DEVICES=0 python3 scripts/semi_crf.py \
#     --input_dir data/datasets/sample \
#     --checkpoint models/semi_crf/Checkpoint_sure_and_possible_True_dataset_mtref_batchsize_1_max_span_size_4_use_transition_layer_False_epoch_2_0.9150.pt \
#     --batchsize 1 

# --- 編集操作の正解ラベル作成 ---
# python3 scripts/generate_label.py --input_dir data/datasets/sample

# --- 制約生成 (oracle) ---
# CUDA_VISIBLE_DEVICES=0 python3 scripts/generate_constraint.py \
#     --input data/datasets/sample/test.jsonl \
#     --src_labels data/datasets/sample/test/test.label.oracle.txt \
#     --dict data/datasets/sample/train.tf.json \
#     --scorer models/roberta_scorer/ \
#     --output data/outputs/generations/sample/test.constraint.oracle.jsonl 

# --- 編集操作予測モデル (訓練) ---
# pip3 install transformers==4.22.2
# CUDA_VISIBLE_DEVICES=0 python3 scripts/edit_label_finetune.py --input_dir data/datasets/sample --model_dir models 

# --- 編集操作予測モデル (予測) ---
# CUDA_VISIBLE_DEVICES=0 python3 scripts/edit_label_predict.py \
#     --input_file data/datasets/sample/test.jsonl --output_file data/datasets/sample/test/test.label.pred.txt \
#     --model_dir  models/edit_label_estimator_sample/version_0 

# --- 制約生成 (pred) ---
# CUDA_VISIBLE_DEVICES=0 python3 scripts/generate_constraint.py \
#     --input data/datasets/sample/test.jsonl \
#     --src_labels data/datasets/sample/test/test.label.pred.txt \
#     --dict data/datasets/sample/train.tf.json \
#     --scorer models/roberta_scorer/ \
#     --output data/outputs/generations/sample/test.constraint.pred.jsonl 
