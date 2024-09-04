export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run_stock_predict_price.py \
  --is_training 1 \
  --root_path dataset/stock/taobao/TSLA \
  --data_config_file dataset/stock/TSLA.json \
  --model_id TSLA_30_30 \
  --model $model_name \
  --data stock_predict_price \
  --features MS \
  --seq_len 30 \
  --pred_len 30 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 5

#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --model_id traffic_96_192 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 192 \
#  --e_layers 4 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.001 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --model_id traffic_96_336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 336 \
#  --e_layers 4 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --d_model 512\
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.001 \
#  --itr 1
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --model_id traffic_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 4 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --batch_size 16 \
#  --learning_rate 0.001\
#  --itr 1
