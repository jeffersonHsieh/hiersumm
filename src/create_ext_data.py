import sentencepiece as spm
import torch
import glob
s = spm.SentencePieceProcessor()
s.Load('/home/lily/af726/multi_news_reproducibility/hiersumm/src/models/spm9998_3.model')
import rouge_papier
from multiprocessing import Pool
import json
import sys
import os
from nltk import tokenize
def compute_doc(num,doc,mode):
    src = [sent for sent in tokenize.sent_tokenize(''.join(doc['src'])) if len(sent.split()) > 3]
    tgt = doc['tgt_str']
    ranks, pairwise_ranks= rouge_papier.compute_extract(src,tgt, mode="sequential", ngram=1,remove_stopwords=True, length=500) #length?
    labels = [1 if r > 0 else 0 for r in ranks]
    #labels = [index for index, label in enumerate(labels) if label == 1]
    if mode == 'test' or mode == 'valid':
        tgt_str = [sent for i,sent in enumerate(src) if labels[i] > 0]
        doc['src_str'] = src
        doc['tgt_str'] = tgt_str
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
    doc['src_sent_labels'] = labels
    print(num, 'processed!')
    return doc
def multi_run_wrapper(args):
    return compute_doc(*args)

def process_file(infile,workers,mode):
    with open(infile) as f:
        docs = json.load(f)
    docs = [(num,doc,mode) for num,doc in enumerate(docs)]
    outname = os.path.basename(infile)[:-5]
    path = f'/home/lily/ch956/hiersumm/src/toys/{outname}'
    if os.path.exists(path):
        print(f'{outname} exists')
        return
    print(outname)
    with Pool(processes = workers) as pool:
        result = pool.map(multi_run_wrapper, docs)
    
    torch.save(result,path)
    
    
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: create_ex_data.py corpus_type portion numworkers')
        exit(1)
    corpus_type = sys.argv[1]
    portion = int(sys.argv[2])
    workers = int(sys.argv[3])
    pts = sorted(glob.glob('/lada2/lily/wikisum_ranked_json/WIKI' + '.' + corpus_type + '.[0-9]*.pt.json'))
    ptslen = len(pts)
    if corpus_type == 'train':
        unit = ptslen//40
        if portion == 39:
            pts = pts[portion*unit:]
        else:
            pts = pts[portion*unit:(portion+1)*unit]
    else:
        if portion == 0:
            pts = pts[:ptslen//3]
        elif portion == 1:
            pts = pts[ptslen//3:2*ptslen//3]
        else:
            pts = pts[2*ptslen//3:]
    print('numfiles:',len(pts))
    for pt in pts:
        process_file(pt,workers,corpus_type)
