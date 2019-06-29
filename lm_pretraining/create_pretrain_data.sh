#!/bin/bash

BERT_BASE_DIR=/PATH/TO/BERT/VOCAB/FILE #modify this to bert or biobert folder containing a vocab.txt file
DATA_DIR=/PATH/TO/TOKENIZED/NOTES #modify this to be the path to the tokenized data 
OUTPUT_DIR=/PATH/TO/OUTPUT/DIR # modify this to be your output directory path


#modify this to be the note type that you want to create pretraining data for - e.g. ecg, echo, radiology, physician, nursing, etc. 
# Note that you can also specify multiple input files & output files below
DATA_FILE=nursing_other 


# Note that create_pretraining_data.py is unmodified from the script in the original BERT repo. 
# Refer to the BERT repo for the most up to date version of this code.
python create_pretraining_data.py \
  --input_file=$DATA_DIR/$DATA_FILE.txt \
  --output_file=$OUTPUT_DIR/$DATA_FILE.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=22 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5