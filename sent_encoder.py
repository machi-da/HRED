import chainer
from chainer import links as L
from chainer import functions as F

class SentEnc(chainer.Chain):
    def __init__(self, hidden, n_layers, dropout):
        super(SentEnc, self).__init__()
        with self.init_scope():
            self.Nlstm = L.NStepLSTM(n_layers, hidden, hidden, dropout)
    
    def __call__(self, hx, cx, xs):
        hy, cy, ys = self.Nlstm(hx, cx, xs)
        return hy, cy, ys