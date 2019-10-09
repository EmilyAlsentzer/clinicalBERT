#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
This code is adapted from the kyzhouhzau/BERT-NER repo with several modifications. 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import collections
import os
import math
import modeling
import optimization
import tokenization
import tensorflow as tf
from tensorflow.python.ops import math_ops
import tf_metrics
import pickle
import numpy as np
import itertools
import json
from random import shuffle
import random

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name", None, "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 1,
                     "How many model checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("cross_val_sz", 10, "Number of cross validation folds")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple token classification."""

    def __init__(self, guid, tokens, text=None, labels=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: (Optional) string. The untokenized text of the sequence. 
          tokens: list of strings. The tokenized sentence. Each token should have a 
            corresponding label for train and dev samples. 
          label: (Optional) list of strings. The labels of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.tokens = tokens
        self.labels = labels

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.guid),
            "tokens: {}".format(" ".join(self.tokens)),
        ]
        if self.text is not None:
            l.append("text: {}".format(self.text))

        if self.labels is not None:
            l.append("labels: {}".format(" ".join(self.labels)))

        return ", ".join(l)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads in data where each line has the word and its corresponding 
        label separated by whitespace. Each sentence is separated by a blank
        line. E.g.:

        Identification  O
        of  O
        APC2    O
        ,   O
        a   O
        homologue   O
        of  O
        the O
        adenomatous B-Disease
        polyposis   I-Disease
        coli    I-Disease
        tumour  I-Disease
        suppressor  O
        .   O

        The O
        adenomatous B-Disease
        polyposis   I-Disease
        ...
        """
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                line = line.strip()
                if len(line) == 0: #i.e. we're in between sentences
                    assert len(words) == len(labels)
                    if len(words) == 0:
                        continue
                    lines.append([words, labels])
                    words = []
                    labels = []
                    continue
                
                word = line.split()[0]
                label = line.split()[-1]
                words.append(word)
                labels.append(label)

            #TODO: see if there's an off by one error here
            return lines

    @classmethod
    def _create_example(self, lines, set_type):
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                words,labels = line
                words = [tokenization.convert_to_unicode(w) for w in words]
                labels = [tokenization.convert_to_unicode(l) for l in labels]
                examples.append(InputExample(guid=guid, tokens=words, labels=labels))
            return examples

    @classmethod
    def _chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def write_cv_to_file(self, evaluation, test, n):
        with open(os.path.join(FLAGS.data_dir, str(n) + '_eval'),'w') as w:
            for example in evaluation:
                for t, l in zip(example.tokens, example.labels):
                    w.write("%s %s\n" %(t, l))
                w.write("\n")
                

        with open(os.path.join(FLAGS.data_dir, str(n) + '_test'),'w') as test_w:
            for test_example in test:
                for t, l in zip(test_example.tokens, test_example.labels):

                    test_w.write("%s %s\n" %(t, l))
                test_w.write("\n")
                
                

    def get_cv_examples(self, splits, n, cv_sz=10):
        # note, when n=9 (10th split), this recovers the original train, dev, test split

        dev = splits[(n-1)%cv_sz] #4 #0 #3 #1
        test = splits[n] #0 #1 #4 #2
        # print('train ind: %d-%d' %((n+1)%cv_sz, (n-1)%cv_sz))
        # print('dev ind: %d' %((n-1)%cv_sz))
        # print('test ind`: %d' %n)
        if (n+1)%cv_sz > (n-1)%cv_sz:
            train = splits[:(n-1)%cv_sz] + splits[(n+1)%cv_sz:]
        else:
            train = splits[(n+1)%cv_sz:(n-1)%cv_sz] #1-3 #2-4 #0-2 #3-0s
        train = list(itertools.chain.from_iterable(train))
        print("Train size: %d, dev size: %d, test size: %d, total: %d" %(len(train), len(dev), len(test), (len(train)+len(dev)+len(test))))
        self.write_cv_to_file(dev, test, n)
        return(train, dev, test)

    def create_cv_examples(self, data_dir, cv_sz=10):
        train_examples = self.get_train_examples(data_dir)
        dev_examples = self.get_dev_examples(data_dir)
        test_examples = self.get_test_examples(data_dir)
        print('num train examples: %d, num eval examples: %d, num test examples: %d' %(len(train_examples), len(dev_examples), len(test_examples)))
        print('Total dataset size: %d' %(len(train_examples) + len(dev_examples) + len(test_examples)))
        random.seed(42)
        train_dev = train_examples + dev_examples
        random.shuffle(train_dev)
        split_sz = math.ceil(len(train_dev)/(cv_sz-1))
        print('Split size: %d' %split_sz)
        splits = list(self._chunks(train_dev, split_sz))
        print('Num splits: %d' %(len(splits) + 1))
        splits = splits + [test_examples]
        print('len splits: ', [len(s) for s in splits])
        return splits



class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_dev.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        test_examples = self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")
        #print(test_examples)
        return test_examples


    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"] 


class NCBIDiseaseProcessor(DataProcessor):
    #https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"] 

    
class i2b22010Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        print('Path: ', os.path.join(data_dir, "test.tsv"))
        test_examples = self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")
        print(test_examples[-5:])
        return test_examples

    def get_labels(self):
        return ["B-problem", "I-problem", "B-treatment", "I-treatment", 'B-test', 'I-test', 'O', "X", "[CLS]", "[SEP]"] 

class i2b22006Processor(DataProcessor):


    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.conll")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.conll")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.conll")), "test"
        )

    def get_labels(self):
        return ["B-ID", "I-ID", "B-HOSPITAL", "I-HOSPITAL", 'B-PATIENT', 'I-PATIENT', 'B-PHONE', 'I-PHONE',
        'B-DATE', 'I-DATE', 'B-DOCTOR', 'I-DOCTOR', 'B-LOCATION', 'I-LOCATION', 'B-AGE', 'I-AGE',
        'O', "X", "[CLS]", "[SEP]"] 

class i2b22012Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        return ['B-OCCURRENCE','I-OCCURRENCE','B-EVIDENTIAL','I-EVIDENTIAL','B-TREATMENT','I-TREATMENT','B-CLINICAL_DEPT',
        'I-CLINICAL_DEPT','B-PROBLEM','I-PROBLEM','B-TEST','I-TEST','O', "X", "[CLS]", "[SEP]"] 

class i2b22014Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        return ["B-IDNUM", "I-IDNUM", "B-HOSPITAL", "I-HOSPITAL", 'B-PATIENT', 'I-PATIENT', 'B-PHONE', 'I-PHONE',
        'B-DATE', 'I-DATE', 'B-DOCTOR', 'I-DOCTOR', 'B-LOCATION-OTHER', 'I-LOCATION-OTHER', 'B-AGE', 'I-AGE', 'B-BIOID', 'I-BIOID',
        'B-STATE', 'I-STATE','B-ZIP', 'I-ZIP', 'B-HEALTHPLAN', 'I-HEALTHPLAN', 'B-ORGANIZATION', 'I-ORGANIZATION',
        'B-MEDICALRECORD', 'I-MEDICALRECORD', 'B-CITY', 'I-CITY', 'B-STREET', 'I-STREET', 'B-COUNTRY', 'I-COUNTRY',
        'B-URL', 'I-URL',
        'B-USERNAME', 'I-USERNAME', 'B-PROFESSION', 'I-PROFESSION', 'B-FAX', 'I-FAX', 'B-EMAIL', 'I-EMAIL', 'B-DEVICE', 'I-DEVICE',
        'O', "X", "[CLS]", "[SEP]"] 

def write_tokens(tokens,tok_to_orig_map, mode, cv_iter):
    #print('MODE: %s' %mode)
    if mode == "test" or mode == "eval":
        path = os.path.join(FLAGS.output_dir, str(cv_iter) + "_token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.write('\n')
        wf.close()
        with open(os.path.join(FLAGS.output_dir, str(cv_iter) + "_tok_to_orig_map_"+mode+".txt"),'a') as w:
            w.write("-1\n") #correspond to [CLS]
            for ind in tok_to_orig_map:
                w.write(str(ind)+'\n')
            w.write("-1\n") #correspond to [SEP]
            w.write('\n')



def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,cv_iter, mode):
    label_map = {}
    for (i, label) in enumerate(label_list,1):
        label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'),'wb') as w:
        pickle.dump(label_map,w)
    orig_tokens = example.tokens
    orig_labels = example.labels
    tokens = []
    labels = []
    tok_to_orig_map = []


    for i, word in enumerate(orig_tokens):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        orig_label = orig_labels[i]
        for m in range(len(token)):
            tok_to_orig_map.append(i)
            if m == 0:
                labels.append(orig_label)
            else:
                labels.append("X")


    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    write_tokens(ntokens,tok_to_orig_map, mode, cv_iter)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file,cv_iter, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, cv_iter, mode )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])

        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities,axis=-1)
        return (loss, per_example_loss, logits, log_probs, predict)
        ##########################################################################
        
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,  per_example_loss, logits, log_probs, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            print('INIT_CHECKPOINT')
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:      
            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids,predictions,num_labels,[1,2],average="macro")
                recall = tf_metrics.recall(label_ids,predictions,num_labels,[1,2],average="macro")
                f = tf_metrics.f1(label_ids,predictions,num_labels,[1,2],average="macro")
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"prediction": predicts, "log_probs": log_probs},
                scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn

def read_tok_file(token_path):
    tokens = list()
    with open(token_path, 'r') as reader:
        for line in reader:
            tok = line.strip()
            if tok == '[CLS]':
                tmp_toks = [tok]
            elif tok == '[SEP]':
                tmp_toks.append(tok)
                tokens.append(tmp_toks)
            elif tok == '':
                continue
            else:
                tmp_toks.append(tok)
    return tokens

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ncbi": NCBIDiseaseProcessor,
        "i2b2_2010": i2b22010Processor,
        "i2b2_2006": i2b22006Processor,
        "i2b2_2014": i2b22014Processor,
        "i2b2_2012": i2b22012Processor
    }
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()


    splits = processor.create_cv_examples(FLAGS.data_dir, cv_sz=FLAGS.cross_val_sz)

    for cv_iter in range(FLAGS.cross_val_sz):
    #for cv_iter in [9]: 
    # if you only want to use the true train, val, test split, then use the last CV split. 
    # We ran out of time to do cross validation so we only used the original train/val/test split.


        tok_eval = os.path.join(FLAGS.output_dir, str(cv_iter) + "_token_eval.txt")
        tok_test = os.path.join(FLAGS.output_dir, str(cv_iter) + "_token_test.txt")

        if os.path.exists(tok_eval):
            os.remove(tok_eval)
        if os.path.exists(tok_test):
            os.remove(tok_test)  
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=FLAGS.keep_checkpoint_max,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None

        if FLAGS.do_train:
            train_examples, eval_examples, test_examples = processor.get_cv_examples(splits, cv_iter, cv_sz=FLAGS.cross_val_sz)
            print('train sz: %d, val size: %d, test size: %d' %(len(train_examples), len(eval_examples), len(test_examples)))
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list)+1,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

        if FLAGS.do_train:
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
            filed_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, cv_iter)
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(train_examples))
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            train_input_fn = file_based_input_fn_builder(
                input_file=train_file,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        if FLAGS.do_eval:
            _, eval_examples, _ = processor.get_cv_examples(splits, cv_iter, cv_sz=FLAGS.cross_val_sz)

            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            filed_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, cv_iter, mode="eval", )

            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d", len(eval_examples))
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
            eval_steps = None
            if FLAGS.use_tpu:
                eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
            eval_drop_remainder = True if FLAGS.use_tpu else False
            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            
            #added
            eval_token_path = os.path.join(FLAGS.output_dir, str(cv_iter) + "_token_eval.txt")
            eval_tokens = read_tok_file(eval_token_path)

            eval_result = estimator.predict(input_fn=eval_input_fn)
            output_predict_file = os.path.join(FLAGS.output_dir, str(cv_iter) + "_label_eval.txt")

            with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'),'rb') as rf:
                label2id = pickle.load(rf)
                id2label = {value:key for key,value in label2id.items()}

            with open(output_predict_file,'w') as p_writer:
                for pidx, prediction in enumerate(eval_result):
                    slen = len(eval_tokens[pidx])
                    output_line = "\n".join(id2label[id] if id!=0 else id2label[3] for id in prediction['prediction'][:slen]) + "\n" #change to O tag
                    p_writer.write(output_line)
                    p_writer.write('\n')
                  


        if FLAGS.do_predict:

            _,_, predict_examples = processor.get_cv_examples(splits, cv_iter, cv_sz=FLAGS.cross_val_sz)

            predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
            filed_based_convert_examples_to_features(predict_examples, label_list,
                                                    FLAGS.max_seq_length, tokenizer,
                                                    predict_file,cv_iter, mode="test")
                                
            tf.logging.info("***** Running prediction*****")
            tf.logging.info("  Num examples = %d", len(predict_examples))
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
            if FLAGS.use_tpu:
                # Warning: According to tpu_estimator.py Prediction on TPU is an
                # experimental feature and hence not supported here
                raise ValueError("Prediction in TPU not supported")
            predict_drop_remainder = True if FLAGS.use_tpu else False
            predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)
            prf = estimator.evaluate(input_fn=predict_input_fn, steps=None)
            tf.logging.info("***** token-level Test evaluation results *****")
            for key in sorted(prf.keys()):
                    tf.logging.info("  %s = %s", key, str(prf[key]))

            test_token_path = os.path.join(FLAGS.output_dir, str(cv_iter) + "_token_test.txt")
            test_tokens = read_tok_file(test_token_path)
   

            result = estimator.predict(input_fn=predict_input_fn)
            output_predict_file = os.path.join(FLAGS.output_dir, str(cv_iter) + "_label_test.txt")

            with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'),'rb') as rf:
                label2id = pickle.load(rf)
                id2label = {value:key for key,value in label2id.items()}

            with open(output_predict_file,'w') as p_writer:
                for pidx, prediction in enumerate(result):
                    slen = len(test_tokens[pidx])
                    output_line = "\n".join(id2label[id] if id!=0 else id2label[3] for id in prediction['prediction'][:slen]) + "\n" #change to O tag
                    p_writer.write(output_line)
                    p_writer.write('\n')
                    

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
