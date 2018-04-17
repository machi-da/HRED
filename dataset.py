import numpy as np
import copy
from collections import Counter
import pickle


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


def load_pickle(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def make_vocab(src, trg, initial_vocab={}, vocabsize=50000, freq=0):
    vocab = copy.copy(initial_vocab)
    word_count = Counter()
    text = []
    for i, (s, t) in enumerate(zip(src, trg)):
        for ss in s:
            text.extend(ss)
        for tt in t:
            text.extend(tt)
        # 10000ごとにCounterへ渡す
        if i % 10000 == 0:
            word_count += Counter(text)
            text = []

    for w in word_count.most_common():
        if w[1] < freq:
            break
        if w[0] not in vocab:
            vocab[w[0]] = len(vocab)
        if len(vocab) == vocabsize:
            break

    return vocab


def convert2label(src, trg, vocab):
    dataset_label = []
    for s, t in zip(src, trg):
        article = [_convert2label(sentence, vocab, vocab['<unk>'], eos=vocab['<eos>']) for sentence in s]
        abstract_sos = [_convert2label(sentence, vocab, vocab['<unk>'], sos=vocab['<sos>']) for sentence in t]
        abstract_eos = [_convert2label(sentence, vocab, vocab['<unk>'], eos=vocab['<eos>']) for sentence in t]
        article[-1][-1] = vocab['<eod>']
        abstract_eos[-1][-1] = vocab['<eod>']
        dataset_label.append((article, abstract_sos, abstract_eos))
    return dataset_label


def _convert2label(words, vocab, unk, sos=None, eos=None):
    word_labels = [vocab[w] if w in vocab else unk for w in words]
    if sos is not None:
        word_labels.insert(0, sos)
    if eos is not None:
        word_labels.append(eos)
    return np.array(word_labels, dtype=np.int32)


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
