
import argparse
import os
import re


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_pred_dir', type=str,  help='Location of input i2b2 prediction formatted files')
parser.add_argument('--input_gold_dir', type=str,  help='Location of input i2b2 gold formatted files')
parser.add_argument('--output_dir', type=str,  help='Location of output files')

args = parser.parse_args()

def format_line(line):
	text, s_line, s_word, e_line, e_word, label = re.findall('c="(.*)" ([0-9]+):([0-9]+) ([0-9]+):([0-9]+)\|\|t="(.*)"', line)[0]
	return ({'text':text, 's_line':s_line, 's_word':s_word, 'e_line':e_line, 'e_word':e_word, 'label':label})
	# Example format: c="cortical-type symptoms" 820:6 820:7||t="Problem"


pred_lines = []
with open(os.path.join(args.input_pred_dir, "i2b2.con"),'r') as pred_f:
	for line in pred_f:
		pred_lines.append(format_line(line))

gold_lines = []
with open(os.path.join(args.input_gold_dir, "i2b2.con"),'r') as gold_f:
	for line in gold_f:
		gold_lines.append(format_line(line))

def in_gold(pred, gold_lines):
	for gold in gold_lines:
		if gold == pred:
			return True
	return False

def in_pred(gold, pred_lines):
	for pred in pred_lines:
		if gold == pred:
			return True
	return False


precCount = 0
for pred in pred_lines:
	if in_gold(pred, gold_lines):
		precCount += 1


recallCount = 0
for gold in gold_lines:
	if in_pred(gold, pred_lines):
		recallCount += 1

  # totalEvents:         total number of Events in the first file
  # recall        total number of Events in the gold file that can be found in the pred file
  # precision        total number of Events in the pred file that can be found in the gold file

systemEventCount = len(pred_lines)
goldEventCount = len(gold_lines)
print('Predicted events: %d, Gold events: %d' %(systemEventCount, goldEventCount))
precision=float(precCount)/systemEventCount
recall=float(recallCount)/goldEventCount
fScore=2*(precision*recall)/(precision+recall)
print('Exact Precision: %0.5f, Recall: %0.5f, F1: %0.5f' %(precision, recall, fScore))

with open(os.path.join(args.output_dir, "final_results.txt"),'w') as writer:
	writer.write('Precision\tRecall\tF1\n')
	writer.write('%0.5f\t%0.5f\t%0.5f\n' %(precision, recall, fScore))

