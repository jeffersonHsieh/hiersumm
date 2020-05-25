import sentencepiece as spm
import torch
s = spm.SentencePieceProcessor()
s.Load('/home/lily/af726/multi_news_reproducibility/hiersumm/src/models/spm9998_3.model')
import rouge_papier
import json
f = open('/lada2/lily/wikisum_ranked_json/WIKI.valid.1.pt.json')
docs = json.load(f)
from nltk import tokenize


for num,doc in enumerate(docs[:2]):
    src = [sent for sent in tokenize.sent_tokenize(''.join(doc['src'])) if len(sent.split()) > 3]
    doc['src_str'] = src
    tgt = doc['tgt_str']
    ranks, pairwise_ranks= rouge_papier.compute_extract(src,tgt, mode="sequential", ngram=1,remove_stopwords=True, length=500) #length?
    labels = [1 if r > 0 else 0 for r in ranks]
    tgt_str = [sent for i,sent in enumerate(src) if labels[i] > 0]
    print(type(tgt_str[0]))
    #labels = [index for index, label in enumerate(labels) if label == 1]
    clss = [0]
    #clss = [-1]*len(src)?
    src = [sent+'</S>' for sent in src]
    sent_len = 0
    src_enc = []
    for sent in src:
        encoded = s.EncodeAsIds(sent)
        sent_len += len(encoded)
        src_enc.extend(encoded)
        clss.append(sent_len-1)
    src_enc = [src_enc[clss[i]:clss[i+1]+1] for i in range(len(clss)-1)]
    doc['src'] = src_enc
    clss.pop(0)
    doc['clss'] = clss
    print('sent_len:', sent_len)
    print(clss)
    print(len(clss))
    doc['src_sent_labels'] = labels
    doc['tgt_str'] = ''.join(tgt_str)
    print(num, 'processed!')
docs = docs[:2]
path = '/home/lily/ch956/hiersumm/src/WIKI.valid.1.pt'
torch.save(docs,path)
