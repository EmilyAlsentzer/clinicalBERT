# Note that this code is adapted from the BioBERT github repo: 
# https://github.com/guidoajansen/biobert/tree/87e70a4dfb0dcc1e29ef9d6562f87c4854504e97/biobert/biocodes

import argparse
import itertools

parser = argparse.ArgumentParser(description='')
parser.add_argument('--token_test_path', type=str,  help='')
parser.add_argument('--label_test_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
parser.add_argument('--output_file', type=str,  help='')
parser.add_argument('--tok_to_orig_map_path', type=str, help='')

args = parser.parse_args()

def detokenize(golden_path, pred_token_test_path, pred_label_test_path, tok_to_orig_map_path, output_file):
    
    """convert word-piece BERT-NER results to original words (CoNLL eval format)
        
    Args:
        golden_path: path to golden dataset. ex) NCBI-disease/test.tsv
        pred_token_test_path: path to token_test.txt from output folder. ex) output/token_test.txt
        pred_label_test_path: path to label_test.txt from output folder. ex) output/label_test.txt
        output_file: file where result will write to. ex) output/conll_format.txt
        
    Outs:
        NER_result_conll.txt
    """
    # read golden
    ans = dict({'toks':[], 'labels':[]})
    with open(golden_path,'r') as in_:
        ans_toks, ans_labels = [],[]
        for line in in_:
            line = line.strip()
            if line == '':
                if len(ans_toks) == 0: # there must be extra empty lines
                    continue
                #ans['toks'].append('[SEP]')
                ans['toks'].append(ans_toks)
                ans['labels'].append(ans_labels)
                ans_toks =[]
                ans_labels=[]
                continue
            tmp = line.split()
            ans_toks.append(tmp[0])
            ans_labels.append(tmp[1])
        if len(ans_toks) > 0: #don't forget the last sentence if there's no final empty line
            ans['toks'].append(ans_toks)
            ans['labels'].append(ans_labels)
    
    # read predicted
    pred = dict({'toks':[], 'labels':[], 'tok_to_orig':[]}) # dictionary for predicted tokens and labels.
    with open(pred_token_test_path,'r') as in_: #'token_test.txt'
        pred_toks = []
        for line in in_:
            line = line.strip()
            if line =='':
                pred['toks'].append(pred_toks)
                pred_toks = []
                continue
            pred_toks.append(line)
        if len(pred_toks) > 0: #don't forget the last sentence if there's no final empty line
            pred['toks'].append(pred_toks)
       
    with open(tok_to_orig_map_path,'r') as in_: #'tok_to_orig_map_test.txt'
        pred_tok_to_orig = []
        for line in in_:
            line = line.strip()
            if line =='':
                pred['tok_to_orig'].append(pred_tok_to_orig)
                pred_tok_to_orig=[]
                continue
            pred_tok_to_orig.append(int(line))
        if len(pred_tok_to_orig) > 0: #don't forget the last sentence if there's no final empty line
            pred['tok_to_orig'].append(pred_tok_to_orig)


    with open(pred_label_test_path,'r') as in_: 
        pred_labels = []
        for line in in_:
            line = line.strip()
            if line in ['[CLS]','[SEP]', 'X']: # replace non-text tokens with O. This will not be evaluated.
                pred_labels.append('O')
                continue
            if line == '':
                pred['labels'].append(pred_labels)
                pred_labels = []
                continue
            pred_labels.append(line)
        if len(pred_labels) > 0: #don't forget the last sentence if there's no final empty line
            pred['labels'].append(pred_labels)
            


    print(len(pred['toks']), len(pred['labels']), len(ans['labels']), len(ans['toks']))
    


    if (len(pred['toks']) != len(pred['labels'])): # Sanity check
        print("Error! : len(pred['toks']) != len(pred['labels']) : Please report us")
        print(len(pred['toks']), len(pred['labels']))
        raise
    
    if (len(ans['labels']) != len(pred['labels'])): # Sanity check
        print(len(ans['labels']), len(pred['labels']))
        print("Error! : len(ans['labels']) != len(bert_pred['labels']) : Please report us")
        raise

    bert_pred = dict({'toks':[], 'labels':[]})
    num_too_short = 0
    for t, l, tok_to_orig_map, ans_toks in zip(pred['toks'],pred['labels'], pred['tok_to_orig'], ans['toks']):
        #remove first and last from each list, which are just buffers
        t.pop(0)
        t.pop()
        l.pop(0)
        l.pop()
        tok_to_orig_map.pop(0)
        tok_to_orig_map.pop()

        if (len(t)!= len(tok_to_orig_map)):
            num_too_short += 1
            print('Sentence of length %d was truncated' %len(tok_to_orig_map))


        bert_pred_toks, bert_pred_labs = [],[]
        for ind_into_orig in range(int(tok_to_orig_map[len(t)-1]) + 1): #indexing into t here to deal with issue of truncated tokens
            tok_indices = [i for i, x in enumerate(tok_to_orig_map) if x == ind_into_orig]
            if len(t) in tok_indices: #skip that token and label because part of the word was truncated during eval
                continue
            wordpiece_toks = [t[ind][2:] if t[ind][:2] == '##' else t[ind] for ind in tok_indices]
            wordpiece_labs = [l[ind] for ind in tok_indices]
            bert_pred_toks.append(''.join(wordpiece_toks))
            bert_pred_labs.append(wordpiece_labs[0])
        if len(ans_toks) != len(bert_pred_toks): #if sentence was truncated assume remaining toks were predicted as 0
            n_missing_labs = len(ans_toks) - len(bert_pred_toks)
            bert_pred_labs.extend(['O'] * n_missing_labs)
        bert_pred['toks'].append(bert_pred_toks)
        bert_pred['labels'].append(bert_pred_labs)


    print('Number of sentences that were truncated: %d' %num_too_short)


    flattened_pred_toks = [item for sublist in bert_pred['toks'] for item in sublist]
    flattened_pred_labs = [item for sublist in bert_pred['labels'] for item in sublist]
    flattened_ans_labs = [item for sublist in ans['labels'] for item in sublist]
    flattened_ans_toks= [item for sublist in ans['toks'] for item in sublist]

    print(len(flattened_pred_toks), len(flattened_pred_labs), len(flattened_ans_labs), len(flattened_ans_toks))


    if (len(bert_pred['toks']) != len(bert_pred['labels'])): # Sanity check
        print("Error! : len(bert_pred['toks']) != len(bert_pred['labels']) : Please report us")
        raise
   
    if (len(ans['labels']) != len(bert_pred['labels'])): # Sanity check
        print("Error! : len(ans['labels']) != len(bert_pred['labels']) : Please report us")
        raise
    
    with open(output_file, 'w') as out_:
        for ans_toks, ans_labs, pred_labs in zip(ans['toks'], ans['labels'], bert_pred['labels']):
            for ans_t, ans_l, pred_l in zip(ans_toks, ans_labs, pred_labs):
                out_.write("%s %s %s\n"%(ans_t, ans_l, pred_l))
            out_.write('\n')


detokenize(args.answer_path, args.token_test_path, args.label_test_path, args.tok_to_orig_map_path, args.output_file)
