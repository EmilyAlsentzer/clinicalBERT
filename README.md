# clinicalBERT
## Repository for Publicly Available Clinical BERT Embeddings Paper (NAACL Clinical NLP Workshop 2019)


### Download Clinical BERT
We are in the process of submitting Clinical BERT to [PhysioNet](https://physionet.org/).

In the interim, the models can be downloaded [here](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=0), or via

```
wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1
```

All models are finetuned from the Cased BERT-Base model. 


To see an example of how to use clinical BERT for the Med NLI tasks, go to the `run_classifier.sh` script in the downstream_tasks folder.

## Contact
Please contact emilya@mit.edu if you have any questions.

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
