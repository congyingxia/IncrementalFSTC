export BATCHSIZE=32
export EPOCHSIZE=5
export SEED=32 #42, 16, 32
export LEARNINGRATE=1e-6

#running time: 45mins per epoch

CUDA_VISIBLE_DEVICES=3 python -u train.Hybrid.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 96 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name base > log.Hybrid.base.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.Hybrid.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 96 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r1 > log.Hybrid.r1.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.Hybrid.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 96 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r2 > log.Hybrid.r2.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 python -u train.Hybrid.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 96 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r3 > log.Hybrid.r3.seed.$SEED.txt 2>&1 &


CUDA_VISIBLE_DEVICES=6 python -u train.Hybrid.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 96 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r4 > log.Hybrid.r4.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u train.Hybrid.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 96 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r5 > log.Hybrid.r5.seed.$SEED.txt 2>&1 &
