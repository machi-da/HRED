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
from evaluate import Evaluate
from hi_seq2seq import HiSeq2SeqModel
from word_encoder import WordEnc
from word_decoder import WordDec
from sent_encoder import SentEnc
from sent_decoder import SentDec


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

    logger.info('[Test start] logging to {}'.format(log_file))
    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    dropout_ratio = float(config['Parameter']['dropout'])
    vocab_type = config['Parameter']['vocab_type']
    """TEST DETAIL"""
    gpu_id = args.gpu
    batch_size = args.batch
    model_file = args.model
    if gpu_id >= 0:
        xp = chainer.cuda.cupy
    else:
        xp = np
    """DATASET"""
    test_src_file = config['Dataset']['test_src_file']
    correct_txt_file = config['Dataset']['correct_txt_file']

    test_data_size = dataset.data_size(test_src_file)
    logger.info('test size: {0}'.format(test_data_size))
    if vocab_type == 'normal':
        vocab = dataset.VocabNormal()
        vocab.load_vocab(model_dir + 'src_vocab.normal.pkl', model_dir + 'trg_vocab.normal.pkl')
        vocab.set_reverse_vocab()
        sos = vocab.src_vocab['<s>']
        eos = vocab.src_vocab['</s>']
        eod = vocab.src_vocab['<eod>']

    elif vocab_type == 'subword':
        vocab = dataset.VocabSubword()
        vocab.load_vocab(model_dir + 'src_vocab.sub.model', model_dir + 'trg_vocab.sub.model')
        sos = vocab.src_vocab.PieceToId('<s>')
        eos = vocab.src_vocab.PieceToId('</s>')
        eod = vocab.src_vocab.PieceToId('<eod>')

    src_vocab_size = len(vocab.src_vocab)
    trg_vocab_size = len(vocab.trg_vocab)
    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    evaluater = Evaluate(correct_txt_file)
    test_iter = iterator.Iterator(test_src_file, test_src_file, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(src_vocab_size, embed_size, hidden_size, dropout_ratio),
        WordDec(trg_vocab_size, embed_size, hidden_size, dropout_ratio),
        SentEnc(hidden_size, dropout_ratio),
        SentDec(hidden_size, dropout_ratio),
        sos, eos, eod)
    chainer.serializers.load_npz(model_file, model)
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TEST"""
    output = []
    for batch in test_iter.generate():
        # batch: (articlesのリスト, abstracts_sosのリスト, abstracts_eosのリスト)タプル
        batch = vocab.convert2label(batch)
        data = converter.convert(batch, gpu_id)
        """
        out: [(sent, attn), (sent, attn), ...] <-バッチサイズ
        sent: decodeされた文のリスト
        attn: 各文のdecode時のattentionのリスト
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            out = model.generate(data[0], data[3])
        output.extend(out)

    res_decode = []
    res_attn = []
    for o in output:
        sent, attn = o
        sentence = dataset.to_list(sent)
        sentence = dataset.eod_truncate(sentence, eod)
        sent_num = len(sentence)
        sentence = [dataset.eos_truncate(s, eos) for s in sentence]
        sentence = [vocab.label2word(s) for s in sentence]
        sentence = dataset.join_sentences(sentence)
        res_decode.append(sentence)
        attn = xp.sum(xp.array(attn[:sent_num]), axis=0) / sent_num
        res_attn.append(attn)

    rank_list = evaluater.rank(res_attn)
    single = evaluater.single(rank_list)
    multiple = evaluater.multiple(rank_list)
    logger.info('single: {} | {}'.format(single[0], single[1]))
    logger.info('multi : {} | {}'.format(multiple[0], multiple[1]))

    with open(model_file + '.hypo_t', 'w')as f:
        [f.write(r + '\n') for r in res_decode]
    with open(model_file + '.attn_t', 'w')as f:
        [f.write('{}\n'.format(r)) for r in res_attn]


if __name__ == '__main__':
    main()