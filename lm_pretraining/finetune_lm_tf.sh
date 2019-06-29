#!/bin/bash

# location of bert or biobert model
# Note that if you use biobert as your base model, you'll need to change init_checkpoint to be biobert_model.ckpt
BERT_BASE_DIR=/PATH/TO/BERT/MODEL

# folder where you want to save your clinical BERT model
OUTPUT_DIR=/PATH/TO/CLINICAL/BERT/OUTPUT/DIR

# folder that contains the tfrecords - this will be the output directory from create_pretrain_data.sh
INPUT_FILES_DIR=/PATH/TO/TFRECORDS

NUM_TRAIN_STEPS=100000 
NUM_WARMUP_STEPS=10000 
LR=5e-5

# This example illustrates the training of Bio+Discharge Summary BERT. If you change the input_file 
# to the tfrecords for all MIMIC sections - e.g.
# --input_file=../data/tf_records/discharge_summary.tfrecord,../data/tf_records/physician.tfrecord,../data/tf_records/nursing.tfrecord,../data/tf_records/nursing_other.tfrecord,../data/tf_records/radiology.tfrecord,../data/tf_records/general.tfrecord,../data/tf_records/respiratory.tfrecord,../data/tf_records/consult.tfrecord,../data/tf_records/nutrition.tfrecord,../data/tf_records/case_management.tfrecord,../data/tf_records/pharmacy.tfrecord,../data/tf_records/rehab_services.tfrecord,../data/tf_records/social_work.tfrecord,../data/tf_records/ecg.tfrecord,../data/tf_records/echo.tfrecord \

  python run_pretraining.py \
  --output_dir=$OUTPUT_DIR \
  --input_file=$INPUT_FILES_DIR/discharge_summary.tfrecord \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=$NUM_TRAIN_STEPS \
  --num_warmup_steps=$NUM_WARMUP_STEPS \
  --learning_rate=$LR \
  --save_checkpoints_steps=50000 \
  --keep_checkpoint_max=15

   