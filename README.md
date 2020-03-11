# clinicalBERT
Repository for [Publicly Available Clinical BERT Embeddings](https://www.aclweb.org/anthology/W19-1909/) (NAACL Clinical NLP Workshop 2019)

## Using Clinical BERT

UPDATE: You can now use ClinicalBERT directly through the [transformers](https://github.com/huggingface/transformers)  library. Check out the [Bio+Clinical BERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) and [Bio+Discharge Summary BERT](https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT) model pages for instructions on how to use the models within the Transformers library. 

## Download Clinical BERT

The Clinical BERT models can also be downloaded [here](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=0), or via

```
wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1
```

`biobert_pretrain_output_all_notes_150000` corresponds to Bio+Clinical BERT, and `biobert_pretrain_output_disch_100000` corresponds to Bio+Discharge Summary BERT. Both models are finetuned from [BioBERT](https://arxiv.org/abs/1901.08746). We specifically use the [BioBERT-Base v1.0 (+ PubMed 200K + PMC 270K)](https://github.com/naver/biobert-pretrained) version of BioBERT.

`bert_pretrain_output_all_notes_150000` corresponds to Clinical BERT, and `bert_pretrain_output_disch_100000` corresponds to Discharge Summary BERT. Both models are finetuned from the cased version of BERT, specifically cased_L-12_H-768_A-12. 

## Reproduce Clinical BERT
#### Pretraining
To reproduce the steps necessary to finetune BERT or BioBERT on MIMIC data, follow the following steps:
1. Run `format_mimic_for_BERT.py` - Note you'll need to change the file paths at the top of the file.
2. Run `create_pretrain_data.sh`
3. Run `finetune_lm_tf.sh`

Note: See issue [#4](https://github.com/EmilyAlsentzer/clinicalBERT/issues/4) for ways to improve section splitting code. 

#### Downstream Tasks
To see an example of how to use clinical BERT for the Med NLI tasks, go to the `run_classifier.sh` script in the downstream_tasks folder. To see an example for NER tasks, go to the `run_i2b2.sh` script.

## Contact
Please post a Github issue or contact emilya@mit.edu if you have any questions.

## Citation
Please acknowledge the following work in papers or derivative software:

Emily Alsentzer, John Murphy, William Boag, Wei-Hung Weng, Di Jin, Tristan Naumann, and Matthew McDermott. 2019. Publicly available clinical BERT embeddings. In Proceedings of the 2nd Clinical Natural Language Processing Workshop, pages 72-78, Minneapolis, Minnesota, USA. Association for Computational Linguistics. 

```
@inproceedings{alsentzer-etal-2019-publicly,
    title = "Publicly Available Clinical {BERT} Embeddings",
    author = "Alsentzer, Emily  and
      Murphy, John  and
      Boag, William  and
      Weng, Wei-Hung  and
      Jin, Di  and
      Naumann, Tristan  and
      McDermott, Matthew",
    booktitle = "Proceedings of the 2nd Clinical Natural Language Processing Workshop",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-1909",
    doi = "10.18653/v1/W19-1909",
    pages = "72--78"
}
```
