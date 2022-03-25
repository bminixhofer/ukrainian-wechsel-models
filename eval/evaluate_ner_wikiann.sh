for seed in 1231 1232 1233 1234 1235 1236 1237 1238
do
    XRT_TPU_CONFIG="localservice;0;localhost:51011" python3 run_ner.py --save_strategy=no --max_seq_length=128 --pad_to_max_length --dataset_name=wikiann --dataset_config_name=uk --model_name_or_path=$1 --text_column_name=tokens --label_column_name=ner_tags --output_dir=temp --overwrite_output_dir --do_train --do_eval --warmup_ratio=0.3 --per_device_train_batch_size=32 --per_device_eval_batch_size=1 --num_train_epochs=5 --evaluation_strategy=epoch --seed=$seed
done