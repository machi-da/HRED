from chainer.dataset.convert import to_device


def convert(batch, gpu_id=None):
    articles = []
    abstracts_sos = []
    abstracts_eos = []
    rule_flag_list = []
    for data in batch:
        articles.append([to_device(gpu_id, sentence) for sentence in data[0]])
        abstracts_sos.append([to_device(gpu_id, sentence) for sentence in data[1]])
        abstracts_eos.append([to_device(gpu_id, sentence) for sentence in data[2]])
        rule_flag_list.append(to_device(gpu_id, data[3]))
    return articles, abstracts_sos, abstracts_eos, rule_flag_list