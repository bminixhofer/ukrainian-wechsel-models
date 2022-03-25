for seed in 1231 1232 1233 1234 1235 1236 1237 1238
do
    XRT_TPU_CONFIG="localservice;0;localhost:51011" python3 run_ner.py --max_seq_length=128 --pad_to_max_length --dataset_name=universal_dependencies --dataset_config_name=uk_iu --model_name_or_path=$1 --text_column_name=tokens --label_column_name=upos --output_dir=temp --overwrite_output_dir --do_train --do_eval --warmup_ratio=0.3 --per_device_train_batch_size=32 --num_train_epochs=10 --seed=$seed
done
