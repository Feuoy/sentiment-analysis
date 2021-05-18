import torch


def prepare_pack_padded_sequence(inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    按照句子长度从大到小排序，

    :param inputs_words: 文本list
    :param seq_lengths: 文本长度list
    :param descending: 是否降序，从大到小
    :return:
    """

    # 【降序文本长度】，【降序文本长度的原序索引】
    """
    排序后tensor，原序的（0、1、2..表示的索引）tensor
    
    torch.sort()，Sorts the elements of the input tensor along a given dimension in ascending order by value.
    descending=True，从大到小
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    # （降序文本长度的原序索引，的升序），【降序文本长度的原序索引，的原序索引（一开始不排序的索引）】
    _, desorted_indices = torch.sort(indices, descending=False)

    # 降序文本：根据文本长度，降序排序文本
    sorted_inputs_words = inputs_words[indices]

    # 降序文本，降序文本长度，一开始不排序的索引
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices
