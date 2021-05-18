import os
import pickle
import numpy as np
from gensim.models import KeyedVectors


def load_pretrained_wordembedding(word_embedding_path):
    """
    加载预训练的词向量，并添加 'PAD'，'UNK'

    word_embedding_path，这个是原始词向量文件
    word_embedding_path + '.pkl'，这个是保存
    word2vec_from_text，这个是对象
    """

    # 如果不存在，“词向量文件的pkl”
    if not os.path.exists(word_embedding_path + '.pkl'):

        # 根据词向量文件，用gensim.models.KeyedVectors，生成词向量隐藏权重矩阵
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.
        """
        word2vec_from_text = KeyedVectors.load_word2vec_format(word_embedding_path, binary=False, encoding='utf-8',
                                                               unicode_errors='ignore')
        # 写成“词向量文件的pkl”
        with open(word_embedding_path + '.pkl', 'wb') as f:
            pickle.dump(word2vec_from_text, f)

    # 存在，直接读取
    else:
        with open(word_embedding_path + '.pkl', 'rb') as f:
            word2vec_from_text = pickle.load(f)

    # 添加PAD、UNK为合适维度的随机权重
    word2vec_from_text.add('PAD', np.random.randn(word2vec_from_text.vector_size))
    word2vec_from_text.add('UNK', np.random.randn(word2vec_from_text.vector_size))

    return word2vec_from_text
