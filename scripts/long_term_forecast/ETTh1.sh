export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_name=TimeGS
gpu=2

seq_len=96
image_height=24
image_width=24
hidden_dim=16
conv_dim=8
p=1


kernel_size=3
ngf=4
n_downsampling=3

ratio=1
extend_len=0
rasterize_type=gaussian

cholesky1=0.4,0.8,1.2
cholesky2=0.0,-0.2,0.2
cholesky3=0.4,0.8,1.2
coefficient=0.5,1

mse_weight=0.5
learning_rate=0.0001
patience=3
batch_size=32


python -u run.py \
  --gpu $gpu \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 2 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks 1 \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period 24,24,24 \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs 10 \
  --patience $patience

python -u run.py \
  --gpu $gpu \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 1 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks 2 \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period 24,24,24,24 \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs 5 \
  --patience $patience

python -u run.py \
  --gpu $gpu \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 1 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks 2 \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period 24 \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs 3 \
  --patience $patience

python -u run.py \
  --gpu $gpu \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 1 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks 1 \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period 24,24,24,24 \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs 1 \
  --patience $patience