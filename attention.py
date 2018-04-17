import chainer
from chainer import links as L
from chainer import functions as F


class Attention(chainer.Chain):
    def __init__(self, hidden):
        super(Attention, self).__init__()
        with self.init_scope():
            #self.Nlstm = L.NStepLSTM(n_layers, hidden, hidden, dropout)
            pass
        self.hidden = hidden
        
    def __call__(self, dec_hs, enc_hs):
        score = F.matmul(dec_hs, enc_hs, False, True)
        align = F.softmax(score, axis=1)
        attention = align.data
        cv = F.matmul(align, enc_hs)

        return cv, attention
