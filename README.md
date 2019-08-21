# clinicalBERT
Repository for [Publicly Available Clinical BERT Embeddings](https://www.aclweb.org/anthology/W19-1909/) (NAACL Clinical NLP Workshop 2019)


## Download Clinical BERT

The Clinical BERT models can be downloaded [here](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=0), or via

```
wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1
```

`biobert_pretrain_output_all_notes_150000` corresponds to Bio+Clinical BERT, and `biobert_pretrain_output_disch_100000` corresponds to Bio+Discharge Summary BERT. Both models are finetuned from [BioBERT](https://arxiv.org/abs/1901.08746). 

## Reproduce Clinical BERT
#### Pretraining
To reproduce the steps necessary to finetune BERT or BioBERT on MIMIC data, follow the following steps:
1. Run `format_mimic_for_BERT.py` - Note you'll need to change the file paths at the top of the file.
2. Run `create_pretrain_data.sh`
3. Run `finetune_lm_tf.sh`

Note: See issue [#4](https://github.com/EmilyAlsentzer/clinicalBERT/issues/4) for ways to improve section splitting code. 

#### Downstream Tasks
To see an example of how to use clinical BERT for the Med NLI tasks, go to the `run_classifier.sh` script in the downstream_tasks folder.

## Contact
Please post a Github issue or contact emilya@mit.edu if you have any questions.

## Citation
Please cite our arXiv paper:
```
@article{alsentzer2019publicly,
  title={Publicly available clinical BERT embeddings},
  author={Alsentzer, Emily and Murphy, John R and Boag, Willie and Weng, Wei-Hung and Jin, Di and Naumann, Tristan and McDermott, Matthew},
  journal={arXiv preprint arXiv:1904.03323},
  year={2019}
}
```
