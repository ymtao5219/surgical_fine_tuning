python code/run.py \
    --model bert-base-multilingual-cased \
    --model_type before_finetuning \
    --probe_set data/sst2.csv \
    --negative_samples data/glue_sentences.csv \
    --seed 42 \
    --num_of_sentences 100 \
    --num_of_negative_batches 5 \
    --test_statistic ks_2samp \
    --alpha 0.01