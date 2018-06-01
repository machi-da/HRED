import chainer
from chainer import links as L
from chainer import functions as F
from attention import Attention


class SentDec(chainer.Chain):
    def __init__(self, hidden, dropout):
        n_layers = 1
        super(SentDec, self).__init__()
        with self.init_scope():
            self.Nlstm = L.NStepLSTM(n_layers, hidden, hidden, dropout)
            self.W_c   = L.Linear(2*hidden, hidden)
            self.attn = Attention()
        self.hidden = hidden
        self.dropout = dropout

    def __call__(self, hx, cx, xs, enc_ys):
        if xs[0] is None:
            zeros = self.xp.zeros((1, self.hidden), dtype=self.xp.float32)
            xs = [chainer.Variable(zeros)]
        hy, cy, ys = self.Nlstm(hx, cx, xs)
        # ysはlistなのでys[0]でVariableを取得
        ys = ys[0]
        cs, attention = self.attn(ys, enc_ys)
        ys = F.tanh(self.W_c(F.dropout(F.concat((cs, ys), axis=1), self.dropout)))
        return hy, cy, ys, attention
