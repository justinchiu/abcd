
function baseline-oracle-intent-kb {
    python main.py \
        --learning-rate $1 \
        --batch-size $2 \
        --epoch 21 --use-intent --use-kb \
        --hidden-dim 768 \
        --model-type bert \
        --wandb-name baseline-bert \
        --prefix 1117 --filename intent_and_kb-lr$1-b$2 --task cds
}


function sweep-baseline-lr1 {
    for lr in 1e-5 2e-5 5e-5; do
        baseline-oracle-intent-kb $lr 64
    done
}

function sweep-baseline-lr2 {
    for lr in 1e-4 2e-4 5e-4; do
        baseline-oracle-intent-kb $lr 64
    done
}

function sweep-baseline-lr3 {
    for lr in 1e-3 2e-3 5e-3; do
        baseline-oracle-intent-kb $lr 64
    done
}
