export BATCHSIZE=16
export EPOCHSIZE=5
export SEED=32 #42, 16, 32
export LEARNINGRATE=1e-6

#running time: 10mins per epoch

CUDA_VISIBLE_DEVICES=0 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name base > log.entailment.base.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r1 > log.entailment.r1.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r2 > log.entailment.r2.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r3 > log.entailment.r3.seed.$SEED.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r4 > log.entailment.r4.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 64 \
    --seed $SEED \
    --round_name r5 > log.entailment.r5.seed.$SEED.txt 2>&1 &
