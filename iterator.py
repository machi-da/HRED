import random


class Iterator:
    def __init__(self, src_file, trg_file, batch_size, sort=True, reverse=False, shuffle=True):
        self.src_file = src_file
        self.trg_file = trg_file
        self.src = None
        self.trg = None
        self.batch_size = batch_size
        self.sort = sort
        # reverse: True -> データの大きい順, False -> 小さい順
        self.reverse = reverse
        self.shuffle = shuffle

    def _set(self):
        self.src = (d for d in open(self.src_file))
        self.trg = (d for d in open(self.trg_file))

    def generate(self, batches_per_sort=10000):
        self._set()
        src, trg = self.src, self.trg
        batch_size = self.batch_size

        data = []
        for x, y in zip(src, trg):
            x_list = []
            x = x.strip().split('|')
            for xx in x:
                x_list.append(xx.split(' '))

            y_list = []
            y = y.strip().split('|')
            for yy in y:
                y_list.append(yy.split(' '))

            data.append([x_list, y_list])

            if len(data) != batch_size * batches_per_sort:
                continue
            if self.sort:
                data = sorted(data, key=lambda x: len(x[1]), reverse=self.reverse)
            batches = [data[b * batch_size : (b + 1) * batch_size]
                       for b in range(batches_per_sort)]

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                yield batch

            data = []

        if len(data) != 0:
            if self.sort:
                data = sorted(data, key=lambda x: len(x[1]), reverse=self.reverse)
            batches = [data[b * batch_size : (b + 1) * batch_size]
                       for b in range(int(len(data) / batch_size) + 1)]

            if self.shuffle:
                random.shuffle(batches)

            for batch in batches:
                # 補足: len(data) == batch_sizeのとき、batchesの最後に空listができてしまうための対策
                if not batch:
                    continue
                yield batch