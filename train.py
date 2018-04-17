#!/usr/bin/env python
import argparse
import os
import sys
import logging
from logging import getLogger
import chainer
import dataset
import converter
import iterator
from hi_seq2seq import HiSeq2SeqModel
from word_encoder import WordEnc
from word_decoder import WordDec
from sent_encoder import SentEnc
from sent_decoder import SentDec


def parse_args():
    parser = argparse.ArgumentParser()
    """model parameters"""
    parser.add_argument('--embed', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    # parser.add_argument('--weightdecay', type=float, default=1.0e-6)
    # parser.add_argument('--gradclip', type=float, default=3.0)
    """train details"""
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--interval', '-i', type=int, default=500)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result/')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    """PARAMETER"""
    embed_size = args.embed
    hidden_size = args.hidden
    n_layers = args.layers
    dropout_ratio = args.dropout
    gpu_id = args.gpu
    n_epoch = args.epoch
    batch_size = args.batch
    interval = args.interval
    out_dir = args.out
    """MKDIR"""
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    """LOGGER"""
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = out_dir + 'log.txt'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('----------')
    logger.info('logging to {0}'.format(out_dir + 'log.txt'))
    """DATASET"""
    base_dir = '/home/lr/machida/yahoo/sentword/'
    train_src = base_dir + 'que_best.txt.sent.word.train'
    train_trg = base_dir + 'ans_best.txt.sent.word.train'
    valid_src = base_dir + 'que_best.txt.sent.word.valid'
    valid_trg = base_dir + 'ans_best.txt.sent.word.valid'

    # base_dir = '/home/lr/machida/yahoo/summarization/'
    # base_dir = '/Users/machida/work/yahoo/'
    # train_src = base_dir + 'que'
    # train_trg = base_dir + 'ans'
    # valid_src = base_dir + 'que'
    # valid_trg = base_dir + 'ans'

    src = dataset.load(train_src)
    trg = dataset.load(train_trg)

    src_valid = dataset.load(valid_src)
    trg_valid = dataset.load(valid_trg)

    # src_valid = src_valid[:10000]
    # trg_valid = trg_valid[:10000]

    # src = src[:100000]
    # trg = trg[:100000]

    logger.info('train size: {0}'.format(len(src)))
    logger.info('valid size: {0}'.format(len(src_valid)))

    logger.info('Build vocabulary')
    init_vocab = {'<unk>': 0, '<sos>': 1, '<eos>': 2, '<sod>': 3, '<eod>': 4}
    vocab = dataset.make_vocab(src, trg, initial_vocab=init_vocab, freq=3)
    dataset.save_pickle(out_dir + 'vocab.pickle', vocab)
    logger.info('vocab size: {0}'.format(len(vocab)))

    train = dataset.convert2label(src, trg, vocab)
    val   = dataset.convert2label(src_valid, trg_valid, vocab)

    train_iter = iterator.Iterator(train, batch_size, padding=True)
    val_iter   = iterator.Iterator(val  , batch_size, repeat=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        WordDec(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        SentEnc(hidden_size, n_layers, dropout_ratio),
        SentDec(hidden_size, n_layers, dropout_ratio),
        vocab['<sos>'], vocab['<eos>'], vocab['<sod>'], vocab['<eod>'])
    """OPTIMIZER"""
    optimizer = chainer.optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    #optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TRAIN"""
    count = 0
    sum_loss = 0
    log = {}
    while train_iter.epoch < n_epoch:
        batch = train_iter.__next__()
        data = converter.convert(batch, gpu_id)
        loss = optimizer.target(*data)
        count += 1
        sum_loss += loss.data
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()
        if train_iter.iteration % interval == 0:  # log per 100 iteration
            logger.info('iteration:{0}, loss:{1}'.format(train_iter.iteration, sum_loss / interval))
            count = 0
            sum_loss = 0
        if train_iter.is_new_epoch:  # log per 1 epoch
            chainer.serializers.save_npz(
                out_dir + 'model_epoch_{0}.npz'.format(train_iter.epoch), model)
            chainer.serializers.save_npz(
                out_dir + 'optimizer_epoch{0}.npz'.format(train_iter.epoch), optimizer)
        """EVALUATE"""
        if train_iter.is_new_epoch:
            val_loss = 0
            val_iter.reset()
            for batch in val_iter:
                data = converter.convert(batch, gpu_id)
                with chainer.using_config('train', False):
                    val_loss += optimizer.target(*data).data
            logger.info('epoch:{0}, val loss:{1}'.format(train_iter.epoch, val_loss))
            log[train_iter.epoch] = val_loss
    """MODEL SAVE"""
    best_epoch = max(log, key=(lambda x: -log[x]))
    logger.info('best_epoch:{0}'.format(best_epoch))
    chainer.serializers.save_npz(out_dir + 'best_model.npz', model)


if __name__ == '__main__':
    main()
