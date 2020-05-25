import rouge_papier
import json
from nltk import tokenize

input_texts = [] # an array of an array of strings (each string is a sentence)
input_path = '/home/lily/af726/projects/spring-2019/summarization_general/data-final/data-extractive-summarization/test.txt.src.tokenized.fixed.cleaned.final.cleaned.truncated'

with open(input_path, 'r') as fr:
	for line in fr:
		# docs = line.split("story_separator_special_tag")
		# for calculating sentences, input docs should be individual sentences
		docs = [sent for sent in tokenize.sent_tokenize(line) if len(sent.split()) > 3] # only return sentences which are of length greater than 3 to avoid any sentence parsing issues
		input_texts.append(docs)
print("len of input_texts: ", len(input_texts))

target_summaries = [] # an array of an array of one string (that string is the target summary)
summary_path = '/home/lily/af726/projects/spring-2019/summarization_general/data-final/data-extractive-summarization/test.txt.tgt.tokenized.fixed.cleaned.final.cleaned.final'

with open(summary_path, 'r') as fr:
	for line in fr:
		summary = [line]
		target_summaries.append(summary)
print("len of target_summaries: ", len(target_summaries))

labels_arr = []

for (docs, summary) in zip(input_texts, target_summaries):
	ranks, pairwise_ranks = rouge_papier.compute_extract(
	        docs, summary, mode="sequential", ngram=1,
	        remove_stopwords=True, length=300) # 300 to match the summary length
	labels = [1 if r > 0 else 0 for r in ranks]
	# print(labels)
	# transform labels variable to indicate indices of the 1s in the labels
	labels = [index for index, label in enumerate(labels) if label == 1]
	# print(labels)

	labels_arr.append(labels)

print("len of labels_arr: ", len(labels_arr))

for i in range(len(input_texts)):
	data = {}
	data["id"] = str(i)
	# data["article"] = tokenize.sent_tokenize(" ".join(input_texts[i])) # an array of strings, each string is a sentence of the documents 
	data["article"] = input_texts[i]
	data["abstract"] = tokenize.sent_tokenize(target_summaries[i][0]) # an array of strings, each string is a sentence of the summary
	data["label"] = labels_arr[i]
	data["source"] = "Multi-News"

	# ensure_ascii=False to unescape unicode characters in json e.g. "\u2013" => "-"
	with open('/home/lily/xm83/ees/ees2/multi_news/test/' + str(i) + '.json', 'w') as outfile:
	    json.dump(data, outfile, indent=4, ensure_ascii=False) # pretty print to the file e.g. "0.json"
	
