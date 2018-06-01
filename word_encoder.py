import chainer
from chainer import links as L
from chainer import functions as F


class WordEnc(chainer.Chain):
    def __init__(self, n_vocab, embed, hidden, dropout):
        n_layers = 1
        super(WordEnc, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed)
            self.Nlstm = L.NStepBiLSTM(n_layers, embed, hidden, dropout)
        self.hidden = hidden

    def __call__(self, hx, cx, xs):
        xs_embed = [self.embed(x) for x in xs]
        hy, cy, ys = self.Nlstm(hx, cx, xs_embed)

        hy = F.sum(hy, axis=0)
        cy = F.sum(cy, axis=0)
        ys = [F.sum(F.reshape(y, (-1, 2, self.hidden)), axis=1) for y in ys]

        return hy, cy, ys