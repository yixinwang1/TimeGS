export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_name=TimeGS
gpu=0

seq_len=96
image_height=24
image_width=24
hidden_dim=8
conv_dim=4

kernel_size=7
n_blocks=2
n_downsampling=3
ngf=4

period=24,24,24

cholesky1=0.8,1.2
cholesky2=0.0
cholesky3=0.4,0.8
coefficient=1

ratio=1
extend_len=0
rasterize_type=gaussian

mse_weight=0.0
learning_rate=0.0001
train_epochs=10
patience=5
batch_size=16


python -u run.py \
  --gpu $gpu \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_${seq_len}_96 \
  --model $model_name \
  --data ETTh2 \
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
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size 7 \
  --period $period \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs 2 \
  --patience $patience

python -u run.py \
  --gpu $gpu \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_${seq_len}_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 2 \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size 7 \
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
  --data_path ETTh2.csv \
  --model_id ETTh2_${seq_len}_336 \
  --model $model_name \
  --data ETTh2 \
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
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --n_blocks $n_blocks \
  --n_downsampling $n_downsampling \
  --kernel_size 7 \
  --period $period \
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
  --data_path ETTh2.csv \
  --model_id ETTh2_${seq_len}_720 \
  --model $model_name \
  --data ETTh2 \
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
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size 3 \
  --period $period \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs 3 \
  --patience $patience