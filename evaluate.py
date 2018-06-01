import copy


class Evaluate:
    def __init__(self, correct_txt_file):
        with open(correct_txt_file, 'r')as f:
            self.correct_data = f.readlines()

    def rank(self, attn_list):
        attn_data = copy.deepcopy(attn_list)
        rank_list = []
        for attn, d in zip(attn_data, self.correct_data):
            label = [int(num) for num in d.split('\t')[0].split(',')]
            rank = []
            for i in range(1, len(attn)+1):
                index = attn.argmax()
                if i in label:
                    rank.append((1, index))
                else:
                    rank.append((0, index))
                attn[index] = -1
            rank_list.append(rank)

        return rank_list

    def single(self, rank_list):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        for r in rank_list:
            sent_num = len(r)
            correct = False
            count = 0
            for rr in r:
                if rr[0] == 1:
                    count += 1
                    if rr[1] == 1:
                        correct = True
            if count == 1:
                score_dic[sent_num][1] += 1
                if correct:
                    score_dic[sent_num][0] += 1
        
        t_correct, t = sum([v[0] for k, v in score_dic.items()]), sum([v[1] for k, v in score_dic.items()])
        for v in score_dic.values():
            if v[1] == 0:
                v[1] = 1
        num = ' '.join([str(round(v[0] / v[1], 3)) for k, v in score_dic.items()]) + ' | {}/{}'.format(t_correct, t)
        rate = ' '.join('{}/{}'.format(v[0], v[1]) for k, v in score_dic.items()) + ' | {}'.format(round(t_correct / t, 3))
        return num, rate

    def multiple(self, rank_list):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        for r in rank_list:
            sent_num = len(r)
            correct_lit = []
            correct = 0
            for i, rr in enumerate(r):
                if rr[0] == 1:
                    correct_lit.append(i)
            for c in correct_lit:
                if r[c][1] in range(1, len(correct_lit) + 1):
                    correct += 1

            score_dic[sent_num][0] += correct
            score_dic[sent_num][1] += len(correct_lit)
        t_correct, t = sum([v[0] for k, v in score_dic.items()]), sum([v[1] for k, v in score_dic.items()])
        for v in score_dic.values():
            if v[1] == 0:
                v[1] = 1
        num = ' '.join([str(round(v[0] / v[1], 3)) for k, v in score_dic.items()]) + ' | {}/{}'.format(t_correct, t)
        rate = ' '.join('{}/{}'.format(v[0], v[1]) for k, v in score_dic.items()) + ' | {}'.format(round(t_correct / t, 3))
        return num, rate