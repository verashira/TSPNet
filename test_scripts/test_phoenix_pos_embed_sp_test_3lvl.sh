#!/bin/bash

set -x -e -u -o pipefail
CHECKPOINT=$(realpath ${CHECKPOINT})

cd ..

DROPOUT=0.4
PATIENCE=8
ENC_LAYERS=1

VOCAB=25000
python test_scripts/test_sign_local.py data-bin/phoenix2014T/sp${VOCAB} \
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
        --save-interval 200 \
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
        --encoder-layers ${ENC_LAYERS} \
        --restore-file ${CHECKPOINT}
