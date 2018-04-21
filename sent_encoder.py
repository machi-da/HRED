import chainer
from chainer import links as L
from chainer import functions as F


class SentEnc(chainer.Chain):
    def __init__(self, hidden, n_layers, dropout, bidirectional):
        super(SentEnc, self).__init__()
        with self.init_scope():
            if bidirectional:
                self.Nlstm = L.NStepBiLSTM(n_layers, hidden, hidden, dropout)
            else:
                self.Nlstm = L.NStepLSTM(n_layers, hidden, hidden, dropout)
        self.hidden_size = hidden
        self.bidirectional = bidirectional
    
    def __call__(self, hx, cx, xs):
        hy, cy, ys = self.Nlstm(hx, cx, xs)

        # bidirectionalの場合はconcatされたベクトルを足し合わせて次元数を合わせている
        if self.bidirectional:
            '''
            hy, cyの処理
            (2*n_layers, batch_size, hidden_size) 
            -> sum: (batch_size, hidden_size)
            -> reshape: (n_layers, batch_size, hidden_size)
            '''
            hy_ = F.reshape(F.sum(hy, axis=0), (1, -1, self.hidden_size))
            cy_ = F.reshape(F.sum(cy, axis=0), (1, -1, self.hidden_size))
            '''
            ysの処理
            (batch_size, 2*hidden_size)
            -> reshape: (batch_size, 2, hidden_size)
            -> sum: (batch_size, hidden_size)
            '''
            ys_ = []
            for y in ys:
                ys_.append(F.sum(F.reshape(y, (-1, 2, self.hidden_size)), axis=1))
            hy, cy, ys = hy_, cy_, ys_

        return hy, cy, ys