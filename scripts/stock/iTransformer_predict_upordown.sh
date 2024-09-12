export CUDA_VISIBLE_DEVICES=0

#model_name=iTransformer
model_name=iTransformer_classify

python -u run_stock_predict_upordown.py \
  --is_training 1 \
  --root_path dataset/stock/all_0.5_up_or_down/TSLA \
  --data_config_file dataset/stock/TSLA.json \
  --model_id TSLA_30_predict_0.5_upordown \
  --model $model_name \
  --data stock_predict_upordown \
  --features MS \
  --num_classes 2 \
  --num_features 4 \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 4 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --train_epochs 5

