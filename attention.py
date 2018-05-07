import chainer
from chainer import links as L
from chainer import functions as F


class Attention:
    def __init__(self):
        super(Attention, self).__init__()

    def __call__(self, dec_hs, enc_hs):
        score = F.matmul(dec_hs, enc_hs, False, True)
        align = F.softmax(score, axis=1)
        attention = align.data
        cv = F.matmul(align, enc_hs)

        return cv, attention