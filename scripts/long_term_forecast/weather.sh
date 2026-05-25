export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_name=TimeGS
gpu=0
device_list=4,5 # Do not include the GPU mentioned in the previous line

seq_len=96
image_height=24
image_width=24
hidden_dim=16
conv_dim=8
p=1

kernel_size=3
ngf=4
n_downsampling=3
n_blocks=2

period=144,144,144,144,144,144
ratio=2
extend_len=0

cholesky1=0.4,0.6,0.8
cholesky2=0.0
cholesky3=0.4,0.6,0.8,1.0,1.2
coefficient=0.5,1

mse_weight=0.5
learning_rate=0.0001
train_epochs=10
patience=5
batch_size=16


python -u run.py \
  --gpu $gpu --device_list $device_list \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_${seq_len}_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 2 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size 5 \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period $period \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience

python -u run.py \
  --gpu $gpu --device_list $device_list \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_${seq_len}_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 4 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period $period \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience

python -u run.py \
  --gpu $gpu --device_list $device_list \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_${seq_len}_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 2 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period $period \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience

python -u run.py \
  --gpu $gpu --device_list $device_list \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_${seq_len}_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p 1 \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --period $period \
  --ratio $ratio \
  --extend_len $extend_len \
  --mse_weight $mse_weight \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience