import torch
from abstractive.model_builder import ExtSummarizer

from abstractive.data_loader import load_dataset
import sentencepiece
spm = sentencepiece.SentencePieceProcessor()
spm.Load('models/spm9998_3.model')

word_padding_idx = spm.PieceToId('<PAD>')
symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': word_padding_idx,
                'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>')}
print(symbols)

vocab_size = len(spm)


class Namespace:
     def __init__(self, **kwargs):
             self.__dict__.update(kwargs)


args = Namespace(accum_count=4, alpha=0, batch_size=10500,beam_size=5, beta1=0.9, beta2=0.998, data_path='ptdata/WIKI', dataset='',dec_dropout = 0.1, dec_hidden_size=256, dec_layers=1, decay_method='noam', emb_size=256, enc_dropout=0.1, enc_hidden_size=256, enc_layers=8, extractive=False, ff_size=1024,gpu_ranks=[0], heads=8, hier=True, inter_heads=8, inter_layers=[6, 7], label_smoothing=0.1, length_penalty='wu', log_file='log.txt', lr=3, max_generator_batches=32, max_grad_norm=0, max_length=250, max_wiki=5,min_length=20, mode='train', model_path='checkpoints/', n_best=1, optim='adam', report_every=100, report_rouge=False, result_path='../../results', save_checkpoint_steps=5000, seed=666, share_decoder_embeddings=True, share_embeddings=True, test_all=False, test_from='../../results', train_from='checkpoints/model_step_100000.pt', train_steps=1000000, trunc_src_nblock=24,trunc_src_ntoken=500, trunc_tgt_ntoken=400, valid_batch_size=10000, visible_gpus='5',warmup_steps=8000, world_size=1)
d_model = 256
args.train_from = 'models/wikisum_model_step_500000.pt'
checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
model = ExtSummarizer(args, word_padding_idx, d_model, vocab_size, 'cuda', checkpoint)
