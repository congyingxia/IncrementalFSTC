export BATCHSIZE=16
export EPOCHSIZE=50
export SEED=32 #42, 16, 32
export LEARNINGRATE=1e-6

#running time: 10mins per epoch

CUDA_VISIBLE_DEVICES=0 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed $SEED \
    --round_name r1 > log.nobase.entailment.r1.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed $SEED \
    --round_name r2 > log.nobase.entailment.r2.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed $SEED \
    --round_name r3 > log.nobase.entailment.r3.seed.$SEED.txt 2>&1 &


CUDA_VISIBLE_DEVICES=5 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed $SEED \
    --round_name r4 > log.nobase.entailment.r4.seed.$SEED.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -u train.entailment.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed $SEED \
    --round_name r5 > log.nobase.entailment.r5.seed.$SEED.txt 2>&1 &
