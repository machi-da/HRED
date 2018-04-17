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


def main():
    """ARGUMENT"""
    parser = argparse.ArgumentParser()
    """model parameters"""
    parser.add_argument('--embed', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    #parser.add_argument('--weightdecay', type=float, default=1.0e-6)
    #parser.add_argument('--gradclip', type=float, default=3.0)
    """train details"""
    parser.add_argument('--batch', '-b', type=int, default=3)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--vocab', '-v', type=str, required=True)
    parser.add_argument('--out', '-o', type=str, default='result/')
    args = parser.parse_args()
    """PARAMETER"""
    embed_size = args.embed
    hidden_size = args.hidden
    n_layers = args.layers
    dropout_ratio = args.dropout
    gpu_id = args.gpu
    batch_size = args.batch
    out_dir = args.out
    vocab_file = args.vocab
    model_file = args.model
    """MKDIR"""
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    """LOGGER"""
    logging.basicConfig(level=logging.DEBUG,\
                        #filename = out_dir + 'log.txt',\
                        format   = "[%(asctime)s] %(message)s",\
                        datefmt  = "%Y/%m/%d %H:%M:%S")
    logger = getLogger("test")
    #logger.info('logging to {0}'.format(out_dir + 'log.txt'))
    """DATASET"""
    # base_dir = '/Users/machida/work/yahoo/'
    # test_src = base_dir + 'que'
    # test_trg = base_dir + 'ans'

    base_dir = '/home/lr/machida/yahoo/bestans/by_number3/'
    test_src = base_dir + 'correct.txt.sentword'
    test_trg = base_dir + 'correct.txt.sentword'
    src = dataset.load(test_src)
    trg = dataset.load(test_trg)
    # src = src[:6]
    # trg = trg[:6]

    # print('src')
    # print(src)


    logger.info('test size: {0}'.format(len(src)))
    vocab = dataset.load_pickle(vocab_file)
    logger.info('vocab size: {0}'.format(len(vocab)))
    test  = dataset.convert2label(src, trg, vocab)
    # print('src convert')
    # print(test)
    test_iter = iterator.Iterator(test, batch_size, repeat=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        WordDec(len(vocab), embed_size, hidden_size, n_layers, dropout_ratio),
        SentEnc(hidden_size, n_layers, dropout_ratio),
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
    golds   = []

    for batch in test_iter:
        data = converter.convert(batch, gpu_id)
        # print('data')
        # print(data)
        # print(data[0])
        # print(type(data[0]))

        a = model(data[0])
        # print('a')
        # print(a)
        # print('a[0]')
        # print(a[0])
        # outputs.append(model(data[0])[0])
        for i, aa in enumerate(a):
            outputs.append(aa)
            golds.append(data[2][i])
        # outputs.append(a[0])
        # golds.append(data[2][0])

    # print('outputs')
    # print(outputs)
    # print('golds')
    # print(golds)

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
        # print(output)
        # print(golds)
        # print('aaa')
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
    # print('_attentino_list')
    # print(_attention_list)
    # print(len(_attention_list))
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