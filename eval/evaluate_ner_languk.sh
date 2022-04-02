for seed in 1 2 3 4 5
do
    XRT_TPU_CONFIG="localservice;0;localhost:51011" python3 run_ner.py --save_strategy=epoch --max_seq_length=128 --pad_to_max_length --dataset_name=preprocess/lang-uk-ner --model_name_or_path=$1 --text_column_name=tokens --label_column_name=ner_tags --output_dir=temp --overwrite_output_dir --do_train --do_eval --warmup_ratio=0.3 --per_device_train_batch_size=16 --per_device_eval_batch_size=2 --num_train_epochs=5 --evaluation_strategy=epoch --seed=$seed --warmup_ratio=0.3 --load_best_model_at_end --metric_for_best_model=f1
done