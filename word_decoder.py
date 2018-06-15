import chainer
from chainer import links as L
from chainer import functions as F


class WordDec(chainer.Chain):
    def __init__(self, n_vocab, embed, hidden, dropout):
        n_layers = 1
        super(WordDec, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed)
            self.Nlstm = L.NStepLSTM(n_layers, embed, hidden, dropout)
            self.proj  = L.Linear(hidden, n_vocab)
        self.dropout = dropout

    def __call__(self, hx, cx, xs):
        xs_embed = [self.embed(x) for x in xs]
        hy, cy, ys = self.Nlstm(hx, cx, xs_embed)
        ys_proj = [self.proj(F.dropout(y, self.dropout)) for y in ys]
        return hy, cy, ys_proj
