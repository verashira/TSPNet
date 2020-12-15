#!/bin/bash

set -x -e -u -o pipefail
cd ..

if [ -z ${SAVE_DIR+x} ]; then 
    echo "no save dir"; 
    PARAM_SAVE_DIR=""
else 
    echo "save dir"
    PARAM_SAVE_DIR="--save-dir ${SAVE_DIR} --maximize-best-checkpoint-metric"
fi

PATIENCE=8
DROPOUT=0.4
ENC_LAYERS=1
WARMUP=0

VOCAB=25000
CUDA_VISIBLE_DEVICES=0 python train.py data-bin/phoenix2014T/sp${VOCAB} \
        --task translation_sign \
        --target-lang de \
        --max-tokens 4096 \
        --num-levels 3 \
        --multilv-args '{"span_lengths": [8, 12, 16], "level_links": [[1, 0], [2, 1], [2, 0]], "stride": 2, "eye": true, "same_level_links": true, "symmetric": true}' \
        --src-lv0-body-feat-root i3d-features/span=8_stride=2 \
        --src-lv1-body-feat-root i3d-features/span=12_stride=2 \
        --src-lv2-body-feat-root i3d-features/span=16_stride=2 \
        --arch transformer_sign \
        --encoder-embed-dim 1024 \
        --decoder-embed-dim 300 \
        --warmup-updates 0 \
        --lr 1e-04 \
        --lr-shrink 0.5 \
        --lr-patience ${PATIENCE} \
        --lr-scheduler reduce_lr_on_plateau \
        --lr-mode max \
        --optimizer adam \
        --activation-fn gelu \
        --criterion label_smoothed_cross_entropy \
        --valid-subset test \
        --label-smoothing 0.1 \
        --weight-decay 0.0001 \
        --dropout ${DROPOUT} \
        --max-epoch 200 \
        --save-interval 1 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 10, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-tokenized-bleu \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-detok space \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --encoder-learned-pos \
        --decoder-learned-pos \
        --decoder-embed-path data-bin/phoenix2014T/sp${VOCAB}/emb \
        --decoder-attention-heads 10 \
        --validate-interval 1 \
        --encoder-layers ${ENC_LAYERS} \
        --no-last-checkpoints \
        --no-epoch-checkpoints \
        ${PARAM_SAVE_DIR}

