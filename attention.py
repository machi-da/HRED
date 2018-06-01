import chainer
from chainer import links as L
from chainer import functions as F


class Attention(chainer.Chain):
    def __init__(self):
        super(Attention, self).__init__()

    def __call__(self, dec_hs, enc_hs):
        """
        dec_hs: (1, hidden) decodeされた隠れ状態
        enc_hs: (文数, den) encodeした隠れ状態
        """
        score = F.matmul(dec_hs, enc_hs, False, True)
        align = F.softmax(score, axis=1)
        # [[score1, score2, ...]]となっているのでdata[0]で次元を落とす
        attention = align.data[0]
        cv = F.matmul(align, enc_hs)

        return cv, attention