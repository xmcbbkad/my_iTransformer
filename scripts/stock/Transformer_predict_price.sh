export CUDA_VISIBLE_DEVICES=0

model_name=Transformer

python -u run_stock_predict_price.py \
  --is_training 1 \
  --root_path dataset/stock/all/TSLA \
  --data_config_file dataset/stock/TSLA.json \
  --model_id TSLA_30_30 \
  --model $model_name \
  --data stock_predict_price \
  --features MS \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 4 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 5

