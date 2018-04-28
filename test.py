#!/usr/bin/env python
import argparse
import configparser
import os
import logging
import numpy as np
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
    """train details"""
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--vocabtype', '-v', choices=['normal', 'subword'], default='normal')
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
    batch_size = args.batch
    gpu_id = args.gpu
    model_file = args.model
    vocab_type = args.vocabtype
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

    logger.info('[Test start]')
    logger.info('logging to {0}'.format(out_dir + 'log.txt'))
    """DATASET"""
    config = configparser.ConfigParser()
    config.read(config_file)

    files = config['Dataset']
    test_src_file = files['test_src_file']
    test_trg_file = files['test_trg_file']

    test_data_size = dataset.data_size(test_src_file)
    logger.info('test size: {0}'.format(test_data_size))
    if vocab_type == 'normal':
        vocab = dataset.VocabNormal()
        vocab.load_vocab(out_dir + 'src_vocab.normal.pkl', out_dir + 'trg_vocab.normal.pkl')
        vocab.set_reverse_vocab()
        sos = vocab.src_vocab['<sos>']
        eos = vocab.src_vocab['<eos>']
        eod = vocab.src_vocab['<eod>']

    elif vocab_type == 'subword':
        vocab = dataset.VocabSubword()
        vocab.load_vocab(out_dir + 'src_vocab.subword.model', out_dir + 'trg_vocab.subword.model')
        sos = vocab.src_vocab.PieceToId('<s>')
        eos = vocab.src_vocab.PieceToId('</s>')
        eod = vocab.src_vocab.PieceToId('<eod>')

    src_vocab_size = len(vocab.src_vocab)
    trg_vocab_size = len(vocab.trg_vocab)
    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    test_iter = iterator.Iterator(test_src_file, test_trg_file, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(src_vocab_size, embed_size, hidden_size, n_layers, dropout_ratio),
        WordDec(trg_vocab_size, embed_size, hidden_size, n_layers, dropout_ratio),
        SentEnc(hidden_size, n_layers, dropout_ratio, bidirectional=bidirectional),
        SentDec(hidden_size, n_layers, dropout_ratio),
        sos, eos, eod)
    chainer.serializers.load_npz(model_file, model)
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TEST"""
    outputs = []
    golds = []

    for batch in test_iter.generate():
        batch = vocab.convert2label(batch)
        data = converter.convert(batch, gpu_id)
        out = model(data[0])

        for i, o in enumerate(out):
            outputs.append(o)
            golds.append(data[2][i])

    def to_list(sentences):
        sentences = [sentence.tolist() for sentence in sentences]
        return sentences

    def eos_truncate(labels, eos_label):
        if eos_label in labels:
            eos_index = labels.index(eos_label)
            labels = labels[:eos_index]
        return labels

    def connect_sentences(sentences):
        sentences = '\t'.join(sentences)
        return sentences

    _outputs = []
    _attention_list = []
    _golds = []
    for output, gold in zip(outputs, golds):
        _attention_list.append(output[1])
        output = to_list(output[0])
        output = [eos_truncate(sentence, eos) for sentence in output]
        output = [eos_truncate(sentence, eod) for sentence in output]
        output = [vocab.label2word(sentence) for sentence in output]
        output = connect_sentences(output)
        _outputs.append(output)
        gold = to_list(gold)
        gold = [eos_truncate(sentence, eos) for sentence in gold]
        gold = [eos_truncate(sentence, eod) for sentence in gold]
        gold = [vocab.label2word(sentence) for sentence in gold]
        gold = connect_sentences(gold)
        _golds.append(gold)

    with open(out_dir + 'hypo.txt', 'w') as f:
        print('\n'.join([sentence for sentence in _outputs]), file=f)
    with open(out_dir + 'refe.txt', 'w') as f:
        print('\n'.join([sentence for sentence in _golds]), file=f)
    with open(out_dir + 'attn.txt', 'w')as f:
        np.set_printoptions(precision=3)
        for i, attn in enumerate(_attention_list, start=1):
            score = None
            for a in attn:
                if score is None:
                    score = a.copy()
                else:
                    score += a
            score /= len(attn)
            print(i, score[0], file=f)


if __name__ == '__main__':
    main()