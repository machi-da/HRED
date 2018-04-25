#!/usr/bin/env python
import argparse
import configparser
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
    parser.add_argument('config_file')
    """model parameters"""
    parser.add_argument('--embed', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--bidirectional', '-bi', action='store_true')
    parser.add_argument('--weightdecay', type=float, default=0.0001)
    parser.add_argument('--gradclip', type=float, default=5.0)
    """train details"""
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--interval', '-i', type=int, default=10000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result/')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file = args.config_file
    """PARAMETER"""
    embed_size = args.embed
    hidden_size = args.hidden
    n_layers = args.layers
    dropout_ratio = args.dropout
    bidirectional = args.bidirectional
    gradclip = args.gradclip
    weightdecay = args.weightdecay
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

    logger.info('[Training start]')
    logger.info('logging to {0}'.format(out_dir + 'log.txt'))
    """DATASET"""
    config = configparser.ConfigParser()
    config.read(config_file)

    files = config['Dataset']
    train_src_file = files['train_src_file']
    train_trg_file = files['train_trg_file']
    valid_src_file = files['valid_src_file']
    valid_trg_file = files['valid_trg_file']

    train_data_size = dataset.data_size(train_trg_file)
    valid_data_size = dataset.data_size(valid_trg_file)
    logger.info('train size: {0}, valid size: {1}'.format(train_data_size, valid_data_size))

    logger.info('Build vocabulary')
    init_vocab = {'<unk>': 0, '<sos>': 1, '<eos>': 2, '<sod>': 3, '<eod>': 4}
    vocab = dataset.make_vocab(train_src_file, train_trg_file, initial_vocab=init_vocab, vocabsize=50000, freq=3)
    dataset.save_pickle(out_dir + 'vocab.pickle', vocab)
    logger.info('vocab size: {0}'.format(len(vocab)))

    train_iter = iterator.Iterator(train_src_file, train_trg_file, batch_size, sort=True, shuffle=False, reverse=True)
    valid_iter = iterator.Iterator(valid_src_file, valid_trg_file, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        WordDec(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        SentEnc(hidden_size, n_layers, dropout_ratio, bidirectional=bidirectional),
        SentDec(hidden_size, n_layers, dropout_ratio),
        vocab['<sos>'], vocab['<eos>'], vocab['<sod>'], vocab['<eod>'])
    """OPTIMIZER"""
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(weightdecay))
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TRAIN"""
    sum_loss = 0
    loss_dic = {}
    for epoch in range(1, n_epoch + 1):
        for i, batch in enumerate(train_iter.generate(), start=1):
            batch = dataset.convert2label(batch, vocab)
            data = converter.convert(batch, gpu_id)
            loss = optimizer.target(*data)
            sum_loss += loss.data
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

            if i % interval == 0:
                logger.info('iteration:{0}, loss:{1}'.format(i, sum_loss))
                sum_loss = 0

        chainer.serializers.save_npz(
            out_dir + 'model_epoch_{0}.npz'.format(epoch), model)
        chainer.serializers.save_npz(
            out_dir + 'optimizer_epoch{0}.npz'.format(epoch), optimizer)

        """EVALUATE"""
        valid_loss = 0
        for batch in valid_iter.generate():
            batch = dataset.convert2label(batch, vocab)
            data = converter.convert(batch, gpu_id)
            with chainer.using_config('train', False):
                valid_loss += optimizer.target(*data).data
        valid_loss /= valid_data_size
        logger.info('epoch:{0}, val loss:{1}'.format(epoch, valid_loss))
        loss_dic[epoch] = valid_loss

    """MODEL SAVE"""
    best_epoch = min(loss_dic, key=(lambda x: loss_dic[x]))
    logger.info('best_epoch:{0}'.format(best_epoch))
    chainer.serializers.save_npz(out_dir + 'best_model.npz', model)


if __name__ == '__main__':
    main()
