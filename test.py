#!/usr/bin/env python
import argparse
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
    """model parameters"""
    parser.add_argument('--embed', type=int, default=216)
    parser.add_argument('--hidden', type=int, default=216)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--bidirectional', '-bi', action='store_true')
    """train details"""
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--vocab', '-v', type=str, required=True)
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
    bidirectional = args.bidirectional
    gpu_id = args.gpu
    batch_size = args.batch
    out_dir = args.out
    vocab_file = args.vocab
    model_file = args.model
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
    base_dir = '/Users/machida/work/yahoo/'
    test_src_file = base_dir + 'que'
    test_trg_file = base_dir + 'ans'

    # base_dir = '/home/lr/machida/yahoo/bestans/by_number3/'
    # test_src_file = base_dir + 'correct.txt.sentword'
    # test_trg_file = base_dir + 'correct.txt.sentword'

    test_data_size = dataset.data_size(test_src_file)
    logger.info('test size: {0}'.format(test_data_size))
    vocab = dataset.load_pickle(vocab_file)
    logger.info('vocab size: {0}'.format(len(vocab)))
    test_iter = iterator.Iterator(test_src_file, test_trg_file, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        WordDec(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        SentEnc(hidden_size, n_layers, dropout_ratio, bidirectional=bidirectional),
        SentDec(hidden_size, n_layers, dropout_ratio),
        vocab['<sos>'], vocab['<eos>'], vocab['<sod>'], vocab['<eod>'],
    )
    chainer.serializers.load_npz(model_file, model)
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TEST"""
    reverse_vocab = {}
    for key, value in vocab.items():
        reverse_vocab[value] = key

    outputs = []
    golds = []

    for batch in test_iter.generate():
        batch = dataset.convert2label(batch, vocab)
        data = converter.convert(batch, gpu_id)
        out = model(data[0])

        for i, o in enumerate(out):
            outputs.append(o)
            golds.append(data[2][i])

    def to_list(sentences):
        sentences = [sentence.tolist() for sentence in sentences]
        return sentences

    def label2word(sentence, vocab):
        sentences = [vocab.get(word, '<unk>') for word in sentence]
        return sentences

    def eos_truncate(labels, eos_label):
        if eos_label in labels:
            eos_index = labels.index(eos_label)
            labels = labels[:eos_index]
        return labels

    def connect_sentences(sentences):
        long_sentence = []
        for sentence in sentences:
            long_sentence.extend(sentence)
        return long_sentence

    _outputs = []
    _attention_list = []
    _golds = []
    for output, gold in zip(outputs, golds):
        _attention_list.append(output[1])
        output = to_list(output[0])
        output = [eos_truncate(sentence, vocab['<eos>']) for sentence in output]
        output = [eos_truncate(sentence, vocab['<eod>']) for sentence in output]
        output = [label2word(sentence, reverse_vocab) for sentence in output]
        output = connect_sentences(output)
        _outputs.append(output)
        gold = to_list(gold)
        gold = [eos_truncate(sentence, vocab['<eos>']) for sentence in gold]
        gold = [eos_truncate(sentence, vocab['<eod>']) for sentence in gold]
        gold = [label2word(sentence, reverse_vocab) for sentence in gold]
        gold = connect_sentences(gold)
        _golds.append(gold)

    with open(out_dir + 'hypo.txt', 'w') as f:
        print('\n'.join([' '.join(sentence) for sentence in _outputs]), file=f)
    with open(out_dir + 'refe.txt', 'w') as f:
        print('\n'.join([' '.join(sentence) for sentence in _golds]), file=f)
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