#!/bin/bash

#update these to the path of the bert model and the path to the NER data
BERT_DIR=/PATH/TO/BERT/MODEL #clinical bert, biobert, or bert

#path to NER data. Make sure to preprocess data according to BIO format first.
#You could use the scripts in the i2b2_preprocessing folder to preprocess the i2b2 data
NER_DIR=/PATH/TO/NER/DATA/DIRECTORY


for EPOCHS in 2 3 4 ; do 
    for LEARN_RATE in 2e-5 3e-5 5e-5 ; do
        for BATCH_SZ in 16 32 ; do 


            OUTPUT_DIR=/PATH/TO/OUTPUT/DIRECTORY #update this to the output directory you want
            mkdir -p $OUTPUT_DIR

            # You can change the task_name to 'i2b2_2014', 'i2b2_2010', 'i2b2_2006', or 'i2b2_2012'
            # Note that you may need to modify the DataProcessor code in `run_ner.py` to adapt to the format of your input
            # If you want to use biobert, change the init_checkpoint to biobert_model.ckpt
            # run_ner.py is adapted from kyzhouhzau's BERT-NER github and the BioBERT repo
            python run_ner.py \
            	--do_train=True \
            	--do_eval=True \
            	--do_predict=True \
                --task_name='i2b2_2006' \
                --vocab_file=$BERT_DIR/vocab.txt \
                --bert_config_file=$BERT_DIR/bert_config.json \
                --init_checkpoint=$BERT_DIR/bert_model.ckpt \
                --num_train_epochs=$EPOCHS \
                --learning_rate=$LEARN_RATE \
                --train_batch_size=$BATCH_SZ \
                --max_seq_length=150 \
                --data_dir=$NER_DIR \
                --output_dir=$OUTPUT_DIR \
                --save_checkpoints_steps=2000


            # Note here we're performing 10 fold CV, but if you want to recover the original train, val, test split, use CV iter = 9
            # Also go to run_ner.py & modify line 738-739 so that you only run the last CV iteration. 
            for CV_ITER in 0 1 2 3 4 5 6 7 8 9 ; do
                for MODE in eval test ; do
                    EVAL_OUTPUT_DIR=$OUTPUT_DIR/$CV_ITER #creates a new folder for each CV iteration
                    mkdir -p $EVAL_OUTPUT_DIR
                    mkdir -p $EVAL_OUTPUT_DIR/$MODE
                    mkdir -p $EVAL_OUTPUT_DIR/$MODE/gold/ 
                    mkdir -p $EVAL_OUTPUT_DIR/$MODE/pred/

                    OUTPUT_FILE=${EVAL_OUTPUT_DIR}/NER_result_conll_${MODE}.txt

                    #convert word-piece BERT NER results to CoNLL eval format
                    #Code is adapted from the BioBERT github
                    python ner_eval/ner_detokenize.py \
                        --token_test_path=${OUTPUT_DIR}/${CV_ITER}_token_${MODE}.txt \
                        --label_test_path=${OUTPUT_DIR}/${CV_ITER}_label_${MODE}.txt \
                        --answer_path=${NER_DIR}/${CV_ITER}_${MODE} \
                        --tok_to_orig_map_path=${OUTPUT_DIR}/${CV_ITER}_tok_to_orig_map_${MODE}.txt \
                        --output_file=$OUTPUT_FILE

                    #convert to i2b2 evaluation format (adapted from Cliner Repo)
                    python ner_eval/format_for_i2b2_eval.py \
                    --results_file $OUTPUT_FILE \
                    --output_gold_dir $EVAL_OUTPUT_DIR/$MODE/gold/ \
                    --output_pred_dir $EVAL_OUTPUT_DIR/$MODE/pred/ 
                    
                    # evaluate performance on i2b2 tasks
                    python ner_eval/score_i2b2.py \
                    --input_gold_dir $EVAL_OUTPUT_DIR/$MODE/gold/ \
                    --input_pred_dir $EVAL_OUTPUT_DIR/$MODE/pred/ \
                    --output_dir $EVAL_OUTPUT_DIR/$MODE
                done
            done
        done
    done
done 



