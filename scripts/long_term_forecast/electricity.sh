export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,roundup_power2_divisions:4

model_name=TimeGS
gpu=2
device_list=3,4,5,6 # Do not include the GPU mentioned in the previous line

seq_len=96
image_height=24
image_width=24
hidden_dim=16
conv_dim=8
p=2

period=24,24,24,24

kernel_size=7
ngf=4
n_downsampling=3
n_blocks=2

ratio=2
extend_len=2
rasterize_type=gaussian

cholesky1=0.4,0.8,1.2
cholesky2=0.0
cholesky3=0.4,0.8,1.2
coefficient=0.5,1

mse_weight=0.5
learning_rate=0.001
train_epochs=10
patience=5
batch_size=16


python -u run.py \
  --gpu $gpu --device_list $device_list \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_${seq_len}_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p $p \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --gpu $gpu \
  --task_name long_term_forecast \
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
  --root_path ../dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_${seq_len}_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p $p \
  --ngf $ngf \
  --n_downsampling $n_downsampling \
  --n_blocks $n_blocks \
  --kernel_size $kernel_size \
  --cholesky1 $cholesky1 \
  --cholesky2 $cholesky2 \
  --cholesky3 $cholesky3 \
  --coefficient $coefficient \
  --gpu $gpu \
  --task_name long_term_forecast \
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
  --root_path ../dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_${seq_len}_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p $p \
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
  --patience $patience \
  --multi_gpu

python -u run.py \
  --gpu $gpu --device_list $device_list \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_${seq_len}_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 321 \
  --itr 1 \
  --image_width $image_width \
  --image_height $image_height \
  --hidden_dim $hidden_dim \
  --conv_dim $conv_dim \
  --p $p \
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
  --patience $patience \
  --multi_gpu
    