import chainer
from chainer import links as L
from chainer import functions as F

class WordEnc(chainer.Chain):
    def __init__(self, n_vocab, embed, hidden, n_layers, dropout):
        super(WordEnc, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed)
            self.Nlstm = L.NStepLSTM(n_layers, embed, hidden, dropout)

    def __call__(self, hx, cx, xs):
        xs_embed = [self.embed(x) for x in xs]
        hy, cy, ys = self.Nlstm(hx, cx, xs_embed)
        return hy, cy, ys
