import argparse
import configparser
import os
import glob
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
from sent_vectorizer import SentVec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_dir = args.model_dir
    """LOAD CONFIG FILE"""
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    assert len(config_files) == 1, 'Put only one config file in the directory'
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)
    """LOGGER"""
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = model_dir + 'log.txt'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('[Test start]')
    logger.info('logging to {0}'.format(log_file))
    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    n_layers = int(config['Parameter']['layers'])
    dropout_ratio = float(config['Parameter']['dropout'])
    bidirectional = config['Parameter'].getboolean('bidirectional')
    vocab_type = config['Parameter']['vocab_type']
    """TEST DETAIL"""
    gpu_id = args.gpu
    batch_size = args.batch
    model_file = args.model
    """DATASET"""
    test_src_file = config['Dataset']['test_src_file']
    test_trg_file = config['Dataset']['test_trg_file']

    test_data_size = dataset.data_size(test_src_file)
    logger.info('test size: {0}'.format(test_data_size))
    if vocab_type == 'normal':
        vocab = dataset.VocabNormal()
        vocab.load_vocab(model_dir + 'src_vocab.normal.pkl', model_dir + 'trg_vocab.normal.pkl')
        vocab.set_reverse_vocab()
        sos = vocab.src_vocab['<sos>']
        eos = vocab.src_vocab['<eos>']
        eod = vocab.src_vocab['<eod>']

    elif vocab_type == 'subword':
        vocab = dataset.VocabSubword()
        vocab.load_vocab(model_dir + 'src_vocab.subword.model', model_dir + 'trg_vocab.subword.model')
        sos = vocab.src_vocab.PieceToId('<s>')
        eos = vocab.src_vocab.PieceToId('</s>')
        eod = vocab.src_vocab.PieceToId('<eod>')

    src_vocab_size = len(vocab.src_vocab)
    trg_vocab_size = len(vocab.trg_vocab)
    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    test_iter = iterator.Iterator(test_src_file, test_trg_file, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(src_vocab_size, embed_size, hidden_size, dropout_ratio, n_layers=n_layers, bidirectional=bidirectional),
        WordDec(trg_vocab_size, embed_size, hidden_size, dropout_ratio, n_layers=1),
        SentEnc(hidden_size, dropout_ratio, n_layers=1),
        SentDec(hidden_size, dropout_ratio, n_layers=1),
        SentVec(hidden_size, dropout_ratio),
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

    for i, batch in enumerate(test_iter.generate(), start=1):
        batch = vocab.convert2label(batch)
        data = converter.convert(batch, gpu_id)
        out = model(data[0])

        for j, o in enumerate(out):
            outputs.append(o)
            golds.append(data[2][j])

        if i % 10 == 0:
            logger.info('Finish: {}'.format(batch_size * i))

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

    with open(model_file + '.hypo.txt', 'w') as f:
        print('\n'.join([sentence for sentence in _outputs]), file=f)
    with open(model_file + '.refe.txt', 'w') as f:
        print('\n'.join([sentence for sentence in _golds]), file=f)
    with open(model_file + '.attn.txt', 'w')as f:
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