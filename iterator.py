import random
import re
import numpy as np

pattern = re.compile(r'\?|？|を教え|か。|サイト|方法|を探し|って|知る|知って|知り')


def rule(sentences):
    point = 0.01
    rule_flag_list = []
    for sentence in sentences:
        sentence = sentence.replace(' ', '')
        if re.search(pattern, sentence):
            rule_flag_list.append(point)
        else:
            rule_flag_list.append(0)
    return np.array([rule_flag_list], dtype=np.float32)


class Iterator:
    def __init__(self, src_file, trg_file, batch_size, sort=True, shuffle=True):
        self.src_file = src_file
        self.trg_file = trg_file
        self.batch_size = batch_size
        self.sort = sort
        self.shuffle = shuffle

    def _load(self, file_name):
        return (d for d in open(file_name))

    def generate(self, batches_per_sort=10000):
        src = self._load(self.src_file)
        trg = self._load(self.trg_file)
        batch_size = self.batch_size

        data = []
        for x, y in zip(src, trg):
            x_list = []
            x_len = 0
            x = x.strip().split('|||')
            rule_flag_list = rule(x)
            # print(rule_flag_list)
            # print(type(rule_flag_list))
            # exit()

            for xx in x:
                sent = xx.split()
                x_len += len(sent)
                x_list.append(sent)

            y_list = []
            y = y.strip().split('|||')
            for yy in y:
                y_list.append(yy.split(' '))

            data.append([x_list, y_list, rule_flag_list, x_len])

            if len(data) != batch_size * batches_per_sort:
                continue
            if self.sort:
                data = sorted(data, key=lambda x: (len(x[0]), x[3]), reverse=True)
            batches = [data[b * batch_size : (b + 1) * batch_size]
                       for b in range(batches_per_sort)]

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                yield batch

            data = []

        if len(data) != 0:
            if self.sort:
                data = sorted(data, key=lambda x: (len(x[0]), x[2]), reverse=True)
            batches = [data[b * batch_size : (b + 1) * batch_size]
                       for b in range(int(len(data) / batch_size) + 1)]

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                # 補足: len(data) == batch_sizeのとき、batchesの最後に空listができてしまうための対策
                if not batch:
                    continue
                yield batch