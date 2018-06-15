import argparse
import configparser
import os
import glob
import logging
from logging import getLogger
import numpy as np
np.set_printoptions(precision=3)

# os.environ["CHAINER_TYPE_CHECK"] = "0"
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
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--interval', '-i', type=int, default=10000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
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

    logger.info('[Training start] logging to {}'.format(log_file))
    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    dropout_ratio = float(config['Parameter']['dropout'])
    weight_decay = float(config['Parameter']['weight_decay'])
    gradclip = float(config['Parameter']['gradclip'])
    vocab_type = config['Parameter']['vocab_type']
    vocab_size = int(config['Parameter']['vocab_size'])
    """TRINING DETAIL"""
    gpu_id = args.gpu
    n_epoch = args.epoch
    batch_size = args.batch
    interval = args.interval
    """DATASET"""
    train_src_file = config['Dataset']['train_src_file']
    train_trg_file = config['Dataset']['train_trg_file']
    valid_src_file = config['Dataset']['valid_src_file']
    valid_trg_file = config['Dataset']['valid_trg_file']
    test_src_file  = config['Dataset']['test_src_file']
    correct_txt_file = config['Dataset']['correct_txt_file']

    train_data_size = dataset.data_size(train_trg_file)
    valid_data_size = dataset.data_size(valid_trg_file)
    logger.info('train size: {0}, valid size: {1}'.format(train_data_size, valid_data_size))

    if vocab_type == 'normal':
        init_vocab = {'<unk>': 0, '<s>': 1, '</s>': 2, '<eod>': 3}
        vocab = dataset.VocabNormal()
        vocab.make_vocab(train_src_file, train_trg_file, init_vocab, vocab_size, freq=0)
        dataset.save_pickle(model_dir + 'src_vocab.pkl', vocab.src_vocab)
        dataset.save_pickle(model_dir + 'trg_vocab.pkl', vocab.trg_vocab)
        sos = vocab.src_vocab['<s>']
        eos = vocab.src_vocab['</s>']
        eod = vocab.src_vocab['<eod>']

    elif vocab_type == 'subword':
        vocab = dataset.VocabSubword()
        if os.path.isfile(model_dir + 'src_vocab.sub.model') and os.path.isfile(model_dir + 'trg_vocab.sub.model'):
            vocab.load_vocab(model_dir + 'src_vocab.sub.model', model_dir + 'trg_vocab.sub.model')
        else:
            vocab.make_vocab(train_trg_file + '.sub', train_trg_file + '.sub', model_dir, vocab_size)
        sos = vocab.src_vocab.PieceToId('<s>')
        eos = vocab.src_vocab.PieceToId('</s>')
        eod = vocab.src_vocab.PieceToId('<eod>')

    src_vocab_size = len(vocab.src_vocab)
    trg_vocab_size = len(vocab.trg_vocab)
    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    train_iter = iterator.Iterator(train_src_file, train_trg_file, batch_size, sort=True, shuffle=True)
    # train_iter = iterator.Iterator(train_src_file, train_trg_file, batch_size, sort=False, shuffle=False)
    valid_iter = iterator.Iterator(valid_src_file, valid_trg_file, batch_size, sort=False, shuffle=False)
    evaluater = Evaluate(correct_txt_file)
    test_iter = iterator.Iterator(test_src_file, test_src_file, batch_size, sort=False, shuffle=False)
    """MODEL"""
    model = HiSeq2SeqModel(
        WordEnc(src_vocab_size, embed_size, hidden_size, dropout_ratio),
        WordDec(trg_vocab_size, embed_size, hidden_size, dropout_ratio),
        SentEnc(hidden_size, dropout_ratio),
        SentDec(hidden_size, dropout_ratio),
        sos, eos, eod)
    """OPTIMIZER"""
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
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
            print(batch)
            exit()
            batch = vocab.convert2label(batch)
            data = converter.convert(batch, gpu_id)
            loss = optimizer.target(*data)
            sum_loss += loss.data
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

            if i % interval == 0:
                logger.info('E{} ## iteration:{}, loss:{}'.format(epoch, i, sum_loss))
                sum_loss = 0
        chainer.serializers.save_npz(model_dir + 'model_epoch_{}.npz'.format(epoch), model)
        # chainer.serializers.save_npz(model_dir + 'optimizer_epoch{0}.npz'.format(epoch), optimizer)

        """EVALUATE"""
        valid_loss = 0
        for batch in valid_iter.generate():
            batch = vocab.convert2label(batch)
            data = converter.convert(batch, gpu_id)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                valid_loss += optimizer.target(*data).data
        logger.info('E{} ## val loss:{}'.format(epoch, valid_loss))
        loss_dic[epoch] = valid_loss

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
            attn = np.sum(np.array(attn[:sent_num]), axis=0) / sent_num
            res_attn.append(attn)

        rank_list = evaluater.rank(res_attn)
        single = evaluater.single(rank_list)
        multiple = evaluater.multiple(rank_list)
        logger.info('E{} ## precision'.format(epoch))
        logger.info('single: {} | {}'.format(single[0], single[1]))
        logger.info('multi : {} | {}'.format(multiple[0], multiple[1]))

        with open(model_dir + 'model_epoch_{}.hypo'.format(epoch), 'w')as f:
            [f.write(r + '\n') for r in res_decode]
        with open(model_dir + 'model_epoch_{}.attn'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(r)) for r in res_attn]
        with open(model_dir + 'model_epoch_{}.prec'.format(epoch), 'w')as f:
            f.write('single\n')
            f.write(single[0] + '\n')
            f.write(single[1] + '\n')
            f.write('multiple\n')
            f.write(multiple[0] + '\n')
            f.write(multiple[1] + '\n')

    """MODEL SAVE"""
    best_epoch = min(loss_dic, key=(lambda x: loss_dic[x]))
    logger.info('best_epoch:{0}'.format(best_epoch))
    chainer.serializers.save_npz(model_dir + 'best_model.npz', model)


if __name__ == '__main__':
    main()
