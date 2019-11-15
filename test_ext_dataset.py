import sentencepiece as spm
s = spm.SentencePieceProcessor()
s.Load('/home/lily/af726/multi_news_reproducibility/hiersumm/src/models/spm9998_3.model')
import rouge_papier
import json
f = open('/lada2/lily/wikisum_ranked_json/WIKI.train.1.pt.json')
docs = json.load(f)
from nltk import tokenize


for doc in docs[:1000]:
    src = [sent for sent in tokenize.sent_tokenize(''.join(doc['src'])) if len(sent.split()) > 3]
    tgt = docs['tgt_str']
    ranks = rouge_papier.compute_extract(src,tgt, mode="sequential", ngram=1,remove_stopwords=True, length=300) #length?
    labels = [1 if r > 0 else 0 for r in ranks]
    #labels = [index for index, label in enumerate(labels) if label == 1]
    src_enc = []
    for sent in src:
        encoded = s.EncodeAsIds(sent)
        src_enc.append(encoded)
    doc['src'] = src_enc
    doc['src_sent_labels'] = labels
