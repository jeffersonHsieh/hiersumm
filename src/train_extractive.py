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

import sentencepiece

from abstractive.model_builder import ExtSummarizer
from abstractive.trainer_ext import build_trainer
from abstractive.predictor_builder import build_predictor
from abstractive.data_loader import load_dataset
from memory.memory import HashingMemory
from memory.utils import bool_flag
import torch
import random

from abstractive import data_loader, model_builder
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



def multi_ext(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i

        procs.append(mp.Process(target=run, args=(args,
            device_id, error_queue), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()

def single_ext(args):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    init_logger(args.log_file)
    if (args.mode == 'train'):
        train(args, device_id)
    elif (args.mode == 'test'):
        step = int(args.test_from.split('.')[-2].split('_')[-1])
        # validate(args, device_id, args.test_from, step)
        test(args, args.test_from, step)
    elif (args.mode == 'validate'):
        wait_and_validate(args, device_id)
    elif (args.mode == 'baseline'):
        baseline()
    elif (args.mode == 'print_flags'):
        print_flags()
    elif (args.mode == 'stats'):
        stats()


def train(args,device_id):
    init_logger(args.log_file)
    logger.info(str(args))

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])

    else:
        checkpoint = None

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': word_padding_idx,
               'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>')}
    print(symbols)
    vocab_size = len(spm)

    #returns a generator
    def train_iter_fct():
        return data_loader.AbstractiveDataloader(args, load_dataset(args, 'train', shuffle=True), symbols, args.batch_size, device,
                                                 shuffle=True, is_test=False)
    model = ExtSummarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)
    logger.info(model)
    trainer = build_trainer(args, device_id, model, optim)

    trainer.train(train_iter_fct, args.train_steps)


def wait_and_validate(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, '*model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        ppl_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            ppl = validate(device_id, cp, step)
            ppl_lst.append((ppl, cp))
            max_step = ppl_lst.index(min(ppl_lst))
            if (i - max_step > 5):
                break
        ppl_lst = sorted(ppl_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(ppl_lst))
        for pp, cp in ppl_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test(args, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, '*model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args,device_id, cp, step)
                    test(args,cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, '*model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)

    #
    # while not os.path.exists(file_path):
    #     time.sleep(1)
    #
    # if os.path.isfile(file_path):
    # # read file
    # else:
    #     raise ValueError("%s isn't a file!" % file_path)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)


    opt = vars(checkpoint['opt'])

    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': word_padding_idx,
               'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>')}

    vocab_size = len(spm)
    model = ExtSummarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    model.eval()

    valid_iter = data_loader.AbstractiveDataloader(args, load_dataset(args, 'valid', shuffle=False), symbols,
                                                   args.batch_size, device, shuffle=False, is_test=False)

    trainer = build_trainer(args, device_id, model, symbols, vocab_size, None)
    stats = trainer.validate(valid_iter)
    trainer._report_step(0, step, valid_stats=stats)
    return stats.ppl()


def test(args, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])

    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': word_padding_idx,
               'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>')}

    vocab_size = len(spm)
    vocab = spm
    model = ExtSummarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    model.eval()

    test_iter = data_loader.AbstractiveDataloader(args, load_dataset(args, args.dataset, shuffle=False), symbols,
                                                  args.valid_batch_size, device, shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)
    #predictor = build_predictor(args, vocab, symbols, model, logger=logger)
    #predictor.translate(test_iter, step)

    # trainer.train(train_iter_fct, valid_iter_fct, FLAGS.train_steps, FLAGS.valid_steps)


def print_flags(args):
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    print(checkpoint['opt'])



def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' %gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        train(args,device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)
