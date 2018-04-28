import numpy as np
import copy
from collections import Counter
import pickle
import sentencepiece as spm


def load(file_name):
    text = []
    with open(file_name)as f:
        data = f.readlines()
    for d in data:
        t = []
        sentences = d.strip().split('|')
        for sentence in sentences:
            t.append(sentence.split(' '))
        text.append(t)
    return text


def data_size(file_name):
    with open(file_name, 'r')as f:
        size = sum([1 for _ in f.readlines()])
    return size


def load_pickle(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def make_vocab(text_file, initial_vocab, vocab_size=10000, freq=0):
    text = load(text_file)

    vocab = copy.copy(initial_vocab)
    word_count = Counter()
    words = []
    for i, t in enumerate(text):
        for tt in t:
            words.extend(tt)
        # 10000ごとにCounterへ渡す
        if i % 10000 == 0:
            word_count += Counter(words)
            words = []
    else:
        word_count += Counter(words)

    for w in word_count.most_common():
        if len(vocab) >= vocab_size:
            break
        if w[1] <= freq:
            break
        if w[0] not in vocab:
            vocab[w[0]] = len(vocab)
    return vocab


class VocabNormal:
    def __init__(self):
        self.src_vocab = None
        self.trg_vocab = None
        self.reverse_vocab = None

    def make_vocab(self, src_file, trg_file, initial_vocab, vocab_size, freq):
        self.src_vocab = make_vocab(src_file, initial_vocab, vocab_size, freq)
        self.trg_vocab = make_vocab(trg_file, initial_vocab, vocab_size, freq)

    def load_vocab(self, src_vocab_file, trg_vocab_file):
        self.src_vocab = load_pickle(src_vocab_file)
        self.trg_vocab = load_pickle(trg_vocab_file)

    def convert2label(self, data):
        src_vocab = self.src_vocab
        trg_vocab = self.trg_vocab

        dataset_label = []
        for d in data:
            src, trg = d[0], d[1]
            src = [self._convert2label(sentence, src_vocab, src_vocab['<unk>'], eos=src_vocab['<eos>']) for sentence in src]
            trg_sos = [self._convert2label(sentence, trg_vocab, trg_vocab['<unk>'], sos=trg_vocab['<sos>']) for sentence in trg]
            trg_eos = [self._convert2label(sentence, trg_vocab, trg_vocab['<unk>'], eos=trg_vocab['<eos>']) for sentence in trg]
            src[-1][-1] = src_vocab['<eod>']
            trg_eos[-1][-1] = trg_vocab['<eod>']
            dataset_label.append((src, trg_sos, trg_eos))
        return dataset_label

    def _convert2label(self, sentence, vocab, unk, sos=None, eos=None):
        word_labels = [vocab[w] if w in vocab else unk for w in sentence]
        if sos is not None:
            word_labels.insert(0, sos)
        if eos is not None:
            word_labels.append(eos)
        return np.array(word_labels, dtype=np.int32)

    def set_reverse_vocab(self):
        reverse_vocab = {}
        for k, v in self.trg_vocab.items():
            reverse_vocab[v] = k
        self.reverse_vocab = reverse_vocab

    def label2word(self, sentence):
        sentence = [self.reverse_vocab.get(word, '<unk>') for word in sentence]
        sentence = ' '.join(sentence)
        return sentence


def make_vocab_sp(text_file, model_name, vocab_size):
    args = '''
            --control_symbols=<eod> 
            --input={} 
            --model_prefix={} 
            --vocab_size={} 
            --hard_vocab_limit=false'''\
        .format(text_file, model_name, vocab_size)
    spm.SentencePieceTrainer.Train(args)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_name + '.model')
    return sp


class VocabSubword:
    def __init__(self):
        self.src_vocab = None
        self.trg_vocab = None

    def make_vocab(self, src_file, trg_file, save_dir, vocab_size):
        self.src_vocab = make_vocab_sp(src_file, save_dir + 'src_vocab.subword', vocab_size)
        self.trg_vocab = make_vocab_sp(trg_file, save_dir + 'trg_vocab.subword', vocab_size)

    def load_vocab(self, src_vocab_file, trg_vocab_file):
        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.Load(src_vocab_file)
        trg_sp.Load(trg_vocab_file)
        self.src_vocab = src_sp
        self.trg_vocab = trg_sp

    def convert2label(self, data):
        src_sp = self.src_vocab
        trg_sp = self.trg_vocab

        dataset_label = []
        for d in data:
            src, trg = d[0], d[1]
            src =     [self._convert2label(' '.join(sentence), src_sp, eos=src_sp.PieceToId('</s>')) for sentence in src]
            trg_sos = [self._convert2label(' '.join(sentence), trg_sp, sos=trg_sp.PieceToId('<s>')) for sentence in trg]
            trg_eos = [self._convert2label(' '.join(sentence), trg_sp, eos=trg_sp.PieceToId('</s>')) for sentence in trg]
            src[-1][-1] = src_sp.PieceToId('<eod>')
            trg_eos[-1][-1] = trg_sp.PieceToId('<eod>')
            dataset_label.append((src, trg_sos, trg_eos))
        return dataset_label

    def _convert2label(self, words, sp, sos=None, eos=None):
        word_labels = sp.EncodeAsIds(words)
        if sos is not None:
            word_labels.insert(0, sos)
        if eos is not None:
            word_labels.append(eos)
        return np.array(word_labels, dtype=np.int32)

    def label2word(self, sentence):
        return self.trg_vocab.DecodeIds(sentence)


if __name__ == '__main__':
    train, val, test = load()
    print(len(train))
    print(len(val))
    print(len(test))
    """
    init_vocab={'<unk>':0, '<sos>':1, '<eos>':2, '<sod>':3, '<eod>':4}
    vocab = make_vocab(train, initial_vocab=init_vocab)
    print(len(vocab))
    with open('vocab.pickle', 'wb') as f:
        pickle.dump(vocab, f)
    """
    vocab = load_pickle('vocab.pickle')
    print(len(vocab))
    train = convert2label(train, vocab)
    val   = convert2label(val,   vocab)
    test  = convert2label(test,  vocab)
    """
    print(train[0][0]) #article
    print(train[0][1]) #abstract(sos)
    print(train[0][2]) #abstract(eos)
    """
