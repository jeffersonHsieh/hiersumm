#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import pickle
import signal
import time


from memory.utils import bool_flag
import torch
import random
from memory.memory import HashingMemory
from train_abstractive import multi_abs,single_abs
from train_extractive import multi_ext,single_ext
from others import distributed
from others.logging import init_logger, logger

model_flags = [ 'emb_size', 'enc_hidden_size', 'dec_hidden_size', 'enc_layers', 'dec_layers', 'block_size', 'heads', 'ff_size', 'hier',
               'inter_layers', 'inter_heads', 'block_size']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_file', default='', type=str)
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-data_path', default='../../data/ranked_abs3_fast_b40/WIKI', type=str)
    parser.add_argument('-model_path', default='../../models', type=str)
    parser.add_argument('-vocab_path', default='../../data/spm9998_3.model', type=str)
    parser.add_argument('-train_from', default='', type=str)

    parser.add_argument('-trunc_src_ntoken', default=500, type=int)
    parser.add_argument('-trunc_tgt_ntoken', default=200, type=int)

    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-enc_layers', default=8, type=int)
    parser.add_argument('-dec_layers', default=1, type=int)
    parser.add_argument('-enc_dropout', default=0.1, type=float)
    parser.add_argument('-dec_dropout', default=0, type=float)
    parser.add_argument('-enc_hidden_size', default=256, type=int)
    parser.add_argument('-dec_hidden_size', default=256, type=int)
    parser.add_argument('-heads', default=8, type=int)
    parser.add_argument('-ff_size', default=1024, type=int)
    parser.add_argument("-hier", type=str2bool, nargs='?',const=True,default=True)


    parser.add_argument('-batch_size', default=10000, type=int)
    parser.add_argument('-valid_batch_size', default=10000, type=int)
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-lr', default=3, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-seed', default=0, type=int)

    parser.add_argument('-train_steps', default=20, type=int)
    parser.add_argument('-save_checkpoint_steps', default=20, type=int)
    parser.add_argument('-report_every', default=100, type=int)


    # multi-gpu
    parser.add_argument('-accum_count', default=1, type=int)
    parser.add_argument('-world_size', default=1, type=int)
    parser.add_argument('-gpu_ranks', default='0', type=str)

    # don't need to change flags
    parser.add_argument("-share_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-share_decoder_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('-max_generator_batches', default=32, type=int)

    # flags for  testing
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('-test_from', default='../../results', type=str)
    parser.add_argument('-result_path', default='../../results', type=str)
    parser.add_argument('-alpha', default=0, type=float)
    parser.add_argument('-length_penalty', default='wu', type=str)
    parser.add_argument('-beam_size', default=5, type=int)
    parser.add_argument('-n_best', default=1, type=int)
    parser.add_argument('-max_length', default=250, type=int)
    parser.add_argument('-min_length', default=20, type=int)
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-dataset', default='', type=str)
    parser.add_argument('-max_wiki', default=5, type=int)

    # flags for  hier
    # flags.DEFINE_boolean('old_inter_att', False, 'old_inter_att')
    parser.add_argument('-inter_layers', default='0', type=str)

    parser.add_argument('-inter_heads', default=8, type=int)
    parser.add_argument('-trunc_src_nblock', default=24, type=int)

    # flags for  graph


    # flags for  learning
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=8000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-label_smoothing', default=0.1, type=float)

    #extractive
    parser.add_argument("--extractive", type=str2bool,nargs='?', default=False,const=True,
                        help="Use an extractive model")
    parser.add_argument("-ext_update_encoder", type=str2bool,nargs='?', const=True,default=False,
                        help="Update parameters in encoder as well")
    #memory layers
    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")
    if parser.parse_known_args()[0].use_memory:
        HashingMemory.register_args(parser)
        parser.add_argument("--mem_enc_positions", type=str, default="",
                            help="Memory positions in the encoder ('4' for inside layer 4, '7,10+' for inside layer 7 and after layer 10)")
        parser.add_argument("--mem_dec_positions", type=str, default="",
                            help="Memory positions in the decoder. Same syntax as `mem_enc_positions`.")


    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    args.inter_layers = [int(i) for i in args.inter_layers.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    if args.extractive:
        if(args.world_size>1):
            multi_ext(args)
        else:
            single_ext(args)
    else:
        if(args.world_size>1):
            multi_abs(args)
        else:
            single_abs(args)
