import chainer
from chainer import links as L
from chainer import functions as F


class SentVec(chainer.Chain):
    def __init__(self, hidden, dropout):
        super(SentVec, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(2*hidden, hidden)
        self.dropout = dropout

    def __call__(self, hy, ys):
        context_vector = []
        # reshape: (1, batch_size, hidden_size) -> (batch_size, hidden_size)
        hy = F.reshape(hy, hy.shape[1:])
        for h, y in zip(hy, ys):
            '''
            最後の隠れ状態h:(1, hidden_size)と各単語y:(単語数, hidden_size)とで要素積をとる
            broadcast: (1, hidden_size) -> (単語数, hidden_size)
            sum: (単語数, hidden_size) -> (hidden_size, )
            '''
            h = F.broadcast_to(h, y.shape)
            m = F.sum(h * y, axis=0)
            context_vector.append(m)

        context_vector = F.stack([v for v in context_vector], axis=0)
        sentence_vector = F.tanh(self.linear(F.dropout(F.concat((hy, context_vector), axis=1), self.dropout)))
        return sentence_vector