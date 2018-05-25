import chainer
from chainer import links as L
from chainer import functions as F


class EndLoop(Exception):
    pass


class HiSeq2SeqModel(chainer.Chain):
    def __init__(self, wordEnc, wordDec, sentEnc, sentDec, sentVec, sos, eos, eod):
        super(HiSeq2SeqModel, self).__init__()
        with self.init_scope():
            self.wordEnc = wordEnc
            self.wordDec = wordDec
            self.sentEnc = sentEnc
            self.sentDec = sentDec
            self.sentVec = sentVec
        self.lossfun = F.softmax_cross_entropy
        self.sos_id = sos
        self.eos_id = eos
        self.eod_id = eod

    def __call__(self, articles, abstracts_sos=None, abstracts_eos=None):
        if abstracts_sos is not None:
            return self.loss(self.forward(articles, abstracts_sos), abstracts_eos)
        else:
            return self.generate(articles)

    def loss(self, b_ys, b_ts):
        y = F.vstack([F.vstack(ys) for ys in b_ys])
        t = F.hstack([F.hstack(ts) for ts in b_ts])
        loss = self.lossfun(y, t)
        return loss

    def forward(self, articles, abstracts):
        hs, cs, enc_ys = self.encode(articles)
        hs = F.transpose(hs, (1, 0, 2))
        cs = F.transpose(cs, (1, 0, 2))
        ys = []
        for h, c, abstract, e in zip(hs, cs, abstracts, enc_ys):
            h = F.transpose(F.reshape(h, (1, *h.shape)), (1, 0, 2))
            c = F.transpose(F.reshape(c, (1, *c.shape)), (1, 0, 2))
            ys.append(self.decode(h, c, abstract, e))
        return ys

    def encode(self, articles):
        """word encoder"""
        sentences = []
        split_num = []
        # articlesの全ての文を一括でencode
        # 文の長さをsplit_numに記録し、sentencesに全ての文のリストを作成
        for article in articles:
            split_num.append(len(article))
            sentences.extend(article)
        # 一括でencode
        word_hy, _, word_ys = self.wordEnc(None, None, sentences)
        # articlesの文数ごとにsentences_vectorにappendしていく
        sentences_vector = []
        start = 0
        for num in split_num:
            sentences_vector.append(word_hy[start:start+num])
            start += num

        # sentences_list.append(self.sentVec(word_hy, word_ys))
        """sentence encoder"""
        sent_hy, sent_cy, sent_ys = self.sentEnc(None, None, sentences_vector)

        # sent encoderの隠れ状態をreturn
        # return sent_hy, sent_cy, sent_ys
        # word encodeの最終状態をreturn
        return sent_hy, sent_cy, sentences_vector

    def decode(self, sent_hs, sent_cs, abstract, enc_ys):
        sentences = []
        pre_sentence = None  # sentDec内部でゼロベクトルへ変換される
        for sentence in abstract:
            """sentence decoder"""
            sent_hs, sent_cs, sent_ys, _ = self.sentDec(sent_hs, sent_cs, [pre_sentence], enc_ys)
            """word decoder"""
            hy, cy, ys = self.wordDec(sent_hs, sent_cs, [sentence])
            pre_sentence = hy[-1]
            sentences.append(ys[0])
        return sentences

    def generate(self, articles, limit_s=7, limit_w=50):
        # 各データをエンコードする(バッチ処理)
        hs, cs, enc_ys = self.encode(articles)

        # 1次元と2次元を入れ替えてバッチ単位にする
        hs = F.transpose(hs, (1, 0, 2))
        cs = F.transpose(cs, (1, 0, 2))

        # 1データずつデコード処理(バッチ処理ではない)
        ys = []
        for h, c, e in zip(hs, cs, enc_ys):
            h = F.transpose(F.reshape(h, (1, *h.shape)), (1, 0, 2))
            c = F.transpose(F.reshape(c, (1, *c.shape)), (1, 0, 2))

            ys.append(self._generate(h, c, e, limit_s, limit_w))
        return ys

    def _generate(self, sent_hs, sent_cs, enc_ys, limit_s, limit_w):
        sentences = []
        attentions = []
        pre_sentence = None  # sentDec内部でゼロベクトルへ変換される
        try:
            for i in range(limit_s):
                """sentence decoder"""
                sent_hs, sent_cs, sent_ys, attention_list = self.sentDec(sent_hs, sent_cs, [pre_sentence], enc_ys)
                attentions.append(attention_list)
                """word decoder"""
                word_hs, word_cs = sent_hs, sent_cs
                sos = self.xp.array([self.sos_id], dtype=self.xp.int32)
                sentence = [sos]
                for j in range(limit_w):
                    word_hs, word_cs, ys = self.wordDec(word_hs, word_cs, [sentence[-1]])
                    word = self.xp.argmax(ys[0].data, axis=1)
                    word = word.astype(self.xp.int32)
                    if word == self.eos_id:
                        break
                    elif word == self.eod_id:
                        raise EndLoop
                    sentence.append(word)
                pre_sentence = word_hs[-1]
                sentence = self.xp.hstack(sentence[1:])
                sentences.append(sentence)
        except EndLoop:
            sentence = self.xp.hstack(sentence[1:])
            sentences.append(sentence)
        return sentences, attentions
