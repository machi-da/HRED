import chainer
from chainer import links as L
from chainer import functions as F


class SentEnc(chainer.Chain):
    def __init__(self, hidden, dropout, n_layers):
        super(SentEnc, self).__init__()
        with self.init_scope():
            self.Nlstm = L.NStepBiLSTM(n_layers, hidden, hidden, dropout)
        self.hidden = hidden

    def __call__(self, hx, cx, xs):
        hy, cy, ys = self.Nlstm(hx, cx, xs)

        '''
        concatされたベクトルを足し合わせて次元数を合わせている
        hy, cyの処理
        (2*n_layers, batch_size, hidden_size) 
        -> sum: (batch_size, hidden_size)
        -> reshape: (n_layers, batch_size, hidden_size)
        '''
        hy = F.reshape(F.sum(hy, axis=0), (1, -1, self.hidden))
        cy = F.reshape(F.sum(cy, axis=0), (1, -1, self.hidden))
        '''
        ysの処理
        (batch_size, 2*hidden_size)
        -> reshape: (batch_size, 2, hidden_size)
        -> sum: (batch_size, hidden_size)
        '''
        ys = [F.sum(F.reshape(y, (-1, 2, self.hidden)), axis=1) for y in ys]

        return hy, cy, ys