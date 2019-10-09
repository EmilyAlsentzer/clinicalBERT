import argparse
import os

# Functions are all taken from Willie Boag's cliner code `documents.py` at 
# https://github.com/text-machine-lab/CliNER/blob/5e1599fb2a2209fa0183f623308516816a033d4f/code/notes/documents.py

def convert_to_i2b2_format(tok_sents, pred_labels, mode=None):
    """
    Purpose: Return the given concept label predictions in i2b2 format

    @param  tokenized_sents. <list-of-lists> of tokenized sentences
    @param  pred_labels.     <list-of-lists> of predicted_labels
    @return                  <string> of i2b2-concept-file-formatted data
    """

    # Return value
    retStr = ''

    concept_tuples = tok_labels_to_concepts(tok_sents, pred_labels, mode)

    # For each classification
    for classification in concept_tuples:

        # Ensure 'none' classifications are skipped
        if classification[0] == 'none':
            raise('Classification label "none" should never happen')

        concept = classification[0]
        lineno  = classification[1]
        start   = classification[2]
        end     = classification[3]

        # A list of words (corresponding line from the text file)
        text = tok_sents[lineno-1]

        #print "\n" + "-" * 80
        #print "classification: ", classification
        #print "lineno:         ", lineno
        #print "start:          ", start
        #print "end             ", end
        #print "text:           ", text
        #print 'len(text):      ', len(text)
        #print "text[start]:    ", text[start]
        #print "concept:        ", concept

        datum = text[start]
        for j in range(start, end):
            datum += " " + text[j+1]
        datum = datum.lower()

        #print 'datum:          ', datum

        # Line:TokenNumber of where the concept starts and ends
        idx1 = "%d:%d" % (lineno, start)
        idx2 = "%d:%d" % (lineno, end)

        # Classification
        label = concept.capitalize()


        # Print format
        retStr +=  "c=\"%s\" %s %s||t=\"%s\"\n" % (datum, idx1, idx2, label)

    # return formatted data
    return retStr.strip()


def tok_labels_to_concepts(tokenized_sents, tok_labels, mode):

    #print tok_labels
    '''
    for gold,sent in zip(tok_labels, tokenized_sents):
        print gold
        print sent
        print
    '''

    # convert 'B-treatment' into ('B','treatment') and 'O' into ('O',None)
    def split_label(label):
        if label == 'O':
            iob,tag = 'O', None
        else:
            #print(label)
            if 'LOCATION-OTHER' in label:
                label = label.replace('LOCATION-OTHER', 'LOCATION_OTHER')
            if len(label.split('-')) != 2:
                print(label.split('-'))
            iob,tag = label.split('-')
        return iob, tag

 

    # preprocess predictions to "correct" starting Is into Bs
    corrected = []
    for lineno,(toks, labels) in enumerate(zip(tokenized_sents, tok_labels)):
        corrected_line = []
        for i in range(len(labels)):
            #'''
            # is this a candidate for error?

            iob,tag = split_label(labels[i])
            if iob == 'I':
                # beginning of line has no previous
                if i == 0:
                    print(mode, 'CORRECTING! A')
                    if mode == 'gold':
                        print(toks, labels)
                    new_label = 'B' + labels[i][1:]
                else:
                    # ensure either its outside OR mismatch type
                    prev_iob,prev_tag = split_label(labels[i-1])
                    if prev_iob == 'O' or prev_tag != tag:
                        print(mode, 'CORRECTING! B')
                        new_label = 'B' + labels[i][1:]
                    else:
                        new_label = labels[i]
            else:
                new_label = labels[i]
            #'''
            corrected_line.append(new_label)
        corrected.append( corrected_line )

    tok_labels = corrected

    concepts = []
    for i,labs in enumerate(tok_labels):

        N = len(labs)
        begins = [ j for j,lab in enumerate(labs) if (lab[0] == 'B') ]

        for start in begins:
            # "B-test"  -->  "-test"
            label = labs[start][1:]

            # get ending token index
            end = start
            while (end < N-1) and tok_labels[i][end+1].startswith('I') and tok_labels[i][end+1][1:] == label:
                end += 1

            # concept tuple
            concept_tuple = (label[1:], i+1, start, end)
            concepts.append(concept_tuple)

    '''
    # test it out
    for i in range(len(tokenized_sents)):
        assert len(tokenized_sents[i]) == len(tok_labels[i])
        for tok,lab in zip(tokenized_sents[i],tok_labels[i]):
            if lab != 'O': print '\t',
            print lab, tok
        print
    exit()
    '''

    # test it out
    test_tok_labels = tok_concepts_to_labels(tokenized_sents, concepts)
    #'''
    for lineno,(test,gold,sent) in enumerate(zip(test_tok_labels, tok_labels, tokenized_sents)):
        for i,(a,b) in enumerate(zip(test,gold)):
            #'''
            if not ((a == b)or(a[0]=='B' and b[0]=='I' and a[1:]==b[1:])):
                print()
                print('lineno:    ', lineno)
                print()
                print('generated: ', test[i-3:i+4])
                print('predicted: ', gold[i-3:i+4])
                print(sent[i-3:i+4])
                print('a[0]:  ', a[0])
                print('b[0]:  ', b[0])
                print('a[1:]: ', a[1:])
                print('b[1:]: ', b[1:])
                print('a[1:] == b[a:]: ', a[1:] == b[1:])
                print()
            #'''
            assert (a == b) or (a[0]=='B' and b[0]=='I' and a[1:]==b[1:])
            i += 1
    #'''
    assert test_tok_labels == tok_labels

    return concepts


def tok_concepts_to_labels(tokenized_sents, tok_concepts):
    # parallel to tokens
    labels = [ ['O' for tok in sent] for sent in tokenized_sents ]

    # fill each concept's tokens appropriately
    for concept in tok_concepts:
        label,lineno,start_tok,end_tok = concept
        labels[lineno-1][start_tok] = 'B-%s' % label
        for i in range(start_tok+1,end_tok+1):
            labels[lineno-1][i] = 'I-%s' % label

    # test it out
    '''
    for i in range(len(tokenized_sents)):
        assert len(tokenized_sents[i]) == len(labels[i])
        for tok,lab in zip(tokenized_sents[i],labels[i]):
            if lab != 'O': print '\t',
            print lab, tok
        print
    exit()
    '''

    return labels



parser = argparse.ArgumentParser(description='')
parser.add_argument('--results_file', type=str,  help='Location of results file in conll format')
parser.add_argument('--output_pred_dir', type=str,  help='Location of where to output prediction formatted files')
parser.add_argument('--output_gold_dir', type=str,  help='Location of where to output gold formatted files')


args = parser.parse_args()

results_dict = {'tokens': [], 'gold_labels': [], 'predicted_labels': []}
with open(args.results_file, 'r') as results: #+'/NER_result_conll.txt'
    toks, gold_labs, pred_labs = [],[],[]
    for line in results:
        line = line.strip()
        if line == '':
            results_dict['tokens'].append(toks)
            results_dict['gold_labels'].append(gold_labs)
            results_dict['predicted_labels'].append(pred_labs)
            toks, gold_labs, pred_labs = [],[],[]
        else:
            tok, gold_lab, pred_lab = line.split()
            toks.append(tok)
            gold_labs.append(gold_lab)
            pred_labs.append(pred_lab)
    if len(toks) > 0:
        results_dict['tokens'].append(toks)
        results_dict['gold_labels'].append(gold_labs)
        results_dict['predicted_labels'].append(pred_labs)

i2b2_format_gold = convert_to_i2b2_format(results_dict['tokens'], results_dict['gold_labels'], 'gold')
i2b2_format_predicted = convert_to_i2b2_format(results_dict['tokens'], results_dict['predicted_labels'], 'pred')

with open(os.path.join(args.output_pred_dir, "i2b2.con"),'w') as writer:
    writer.write(i2b2_format_predicted)

with open(os.path.join(args.output_gold_dir, "i2b2.con"),'w') as writer:
    writer.write(i2b2_format_gold)


