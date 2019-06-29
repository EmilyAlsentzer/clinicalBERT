import psycopg2
import pandas as pd
import sys
import spacy
import re
import stanfordnlp
import time
import scispacy
from tqdm import tqdm
from heuristic_tokenize import sent_tokenize_rules 


# update these constants to run this script
OUTPUT_DIR = '/PATH/TO/OUTPUT/DIR' #this path will contain tokenized notes. This dir will be the input dir for create_pretrain_data.sh
MIMIC_NOTES_FILE = 'PATH/TO/MIMIC/DATA' #this is the path to mimic data if you're reading from a csv. Else uncomment the code to read from database below


#setting sentence boundaries
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

#convert de-identification text into one token
def fix_deid_tokens(text, processed_text):
    deid_regex  = r"\[\*\*.{0,15}.*?\*\*\]" 
    if text:
        indexes = [m.span() for m in re.finditer(deid_regex,text,flags=re.IGNORECASE)]
    else:
        indexes = []
    for start,end in indexes:
        processed_text.merge(start_idx=start,end_idx=end)
    return processed_text
    

def process_section(section, note, processed_sections):
    # perform spacy processing on section
    processed_section = nlp(section['sections'])
    processed_section = fix_deid_tokens(section['sections'], processed_section)
    processed_sections.append(processed_section)

def process_note_helper(note):
    # split note into sections
    note_sections = sent_tokenize_rules(note)
    processed_sections = []
    section_frame = pd.DataFrame({'sections':note_sections})
    section_frame.apply(process_section, args=(note,processed_sections,), axis=1)
    return(processed_sections)

def process_text(sent, note):
    sent_text = sent['sents'].text
    if len(sent_text) > 0 and sent_text.strip() != '\n':
        if '\n' in sent_text:
            sent_text = sent_text.replace('\n', ' ')
        note['text'] += sent_text + '\n'  

def get_sentences(processed_section, note):
    # get sentences from spacy processing
    sent_frame = pd.DataFrame({'sents': list(processed_section['sections'].sents)})
    sent_frame.apply(process_text, args=(note,), axis=1)

def process_note(note):
    try:
        note_text = note['text'] #unicode(note['text'])
        note['text'] = ''
        processed_sections = process_note_helper(note_text)
        ps = {'sections': processed_sections}
        ps = pd.DataFrame(ps)
        ps.apply(get_sentences, args=(note,), axis=1)
        return note 
    except Exception as e:
        pass
        #print ('error', e)



if len(sys.argv) < 2:
        print('Please specify the note category.')
        sys.exit()

category = sys.argv[1]


start = time.time()
tqdm.pandas()

print('Begin reading notes')


# Uncomment this to use postgres to query mimic instead of reading from a file
# con = psycopg2.connect(dbname='mimic', host="/var/run/postgresql")
# notes_query = "(select * from mimiciii.noteevents);"
# notes = pd.read_sql_query(notes_query, con)
notes = pd.read_csv(MIMIC_NOTES_FILE, index_col = 0)
#print(set(notes['category'])) # all categories


notes = notes[notes['category'] == category]
print('Number of notes: %d' %len(notes.index))
notes['ind'] = list(range(len(notes.index)))

# NOTE: `disable=['tagger', 'ner'] was added after paper submission to make this process go faster
# our time estimate in the paper did not include the code to skip spacy's NER & tagger
nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])
nlp.add_pipe(sbd_component, before='parser')  


formatted_notes = notes.progress_apply(process_note, axis=1)
with open(OUTPUT_DIR  + category + '.txt','w') as f:
    for text in formatted_notes['text']:
        if text != None and len(text) != 0 :
            f.write(text)
            f.write('\n')

end = time.time()
print (end-start)
print ("Done formatting notes")














