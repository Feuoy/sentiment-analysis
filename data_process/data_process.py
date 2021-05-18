import pandas as pd
import numpy as np
from tqdm import tqdm
import jieba
import pickle
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from base.base_dataset import NLPDataSet, WordEmbedding
from utils.data_process_utils import load_pretrained_wordembedding
import json


class HotspringExample():
    """
    领域数据集实例
    """

    def __init__(self, text, label):
        """
        如果是预测，数据无label，可以设置label=None，根据需求对collate_fn_4_inference部分进行适当修改
        """
        self.text = text
        self.tokens = []
        self.tokens_ids = []
        self.label = label


class HotspringDataSet(NLPDataSet):
    """
    领域数据集构建器实例，继承NLPDataset
    """

    def __init__(self, data_dir, data_name, device, test_size=0.3, bert_path=None,
                 is_mlp=False, needed_by_mlp_max_seq_len=260, word_embedding_path=None):
        """

        :param data_dir: 数据集目录
        :param data_name: 数据集文件名
        :param word_embedding_path: 预训练词向量路径
        :param device: 计算代理
        :param test_size:
        测试集比例，默认0.3
        :param bert_path:  bert预训练模型路径
        """

        self.test_size = test_size
        self.device = device
        self.data_dir = data_dir
        self.data_name = data_name
        self.word_embedding_path = word_embedding_path
        self.is_mlp = is_mlp
        self.needed_by_mlp_max_seq_len = needed_by_mlp_max_seq_len

        # bert训练
        if bert_path:
            # 加载bert自己的tokenizer（相当于词向量已实现）
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            # 加载源数据集
            self.data = self._load_dataset_4_bert()
            # 为了保持model输入的统一
            self.word_embedding = None

        # 普通训练
        else:
            # 加载源数据集、预训练词向量
            self.data, self.word_embedding = self._load_dataset()

        # 分割训练集、测试集
        """
        sklearn.model_selection.train_test_split()，
        Split arrays or matrices into random train and test subsets
        """
        self.train_set, self.test_set = train_test_split(self.data, test_size=self.test_size)

    def __getitem__(self, index):
        """
        获取一个item
        """

        return self.data[index]

    def __len__(self):
        """
        获取长度
        """

        return len(self.data)

    def _load_dataset(self):
        """
        加载源数据集，并构建词嵌入

        :return: 数据集，词嵌入
        """

        # 数据集文件名
        file_name = self.data_name.split('.')[0]

        # 数据集pkl，不存在
        # 1.加载训练集、构建训练集上的词汇表
        # 2.构建训练集上的word embedding
        if not self.is_mlp:
            examples_path = os.path.join(self.data_dir, file_name + '_examples.pkl')
        else:
            examples_path = os.path.join(self.data_dir, file_name + '_examples_mlp.pkl')

        if not os.path.exists(examples_path):

            # 1、加载预训练词向量文件
            pretrained_wordembedding = load_pretrained_wordembedding(self.word_embedding_path)

            # 2、初始化词汇表、词向量
            """
            stoi: 字典，token键，index值；'UNK': 0
            itos: 也是字典，index键，token值；0: 'UNK'
            vectors: 词向量集合
            word_embedding id:      -1.08563066e+00     9.97345448e-01      ......
            """
            stoi = {}
            itos = {}
            stoi['UNK'] = 0
            stoi['PAD'] = 1
            itos[0] = 'UNK'
            itos[1] = 'PAD'
            vectors = []
            vectors.append(pretrained_wordembedding['UNK'])
            vectors.append(pretrained_wordembedding['PAD'])

            # 3、根据词汇表、词向量，构建数据集实例

            # 数据集
            examples = []

            # 读取源数据集csv，为raw_data
            """
            header，数据开始行数
            names，列名列表
            """
            raw_data = pd.read_csv(os.path.join(self.data_dir, self.data_name), header=0, names=['label', 'text'])

            # encoding = 'utf-8', dtype = str
            # .astype(str)

            # 遍历raw_data.iterrows()，作item，行迭代
            for item in tqdm(raw_data.iterrows()):

                # new一个数据集对象
                hotspring_example = HotspringExample(item[1]['text'], item[1]['label'])

                # 使用词向量
                """
                分词，
                jieba.lcut()，返回列表
                jieba.cut()，返回迭代器
                """
                hotspring_example.tokens = [*jieba.lcut(str(hotspring_example.text))]

                # 使用字向量
                """
                直接转成列表
                """
                # hotspring_example.tokens = list(hotspring_example.text)

                # 遍历文本的tokens，为token
                for token in hotspring_example.tokens:

                    # 如果token在预训练词向量中
                    if token in pretrained_wordembedding:

                        # 如果不在stoi，加到词汇表
                        if token not in stoi:
                            # stoi、itos———— 放进
                            stoi[token] = len(stoi)
                            itos[len(stoi)] = token
                            # 根据token找pretrained_wordembedding对象中对应的一条词向量，放到vectors，用于构造word_embedding对象
                            vectors.append(pretrained_wordembedding[token])

                        # hotspring_example对象的tokens_ids（list），，根据{'UNK' : 0}添加0
                        hotspring_example.tokens_ids.append(stoi[token])

                    # 如果token不在
                    else:
                        # hotspring_example对象的tokens_ids（list），，直接添加0
                        hotspring_example.tokens_ids.append(stoi['UNK'])

                # 如果是mlp，padding到260
                """
                    也可以在mlp_model的时候再补齐的，
                    这样就做到和其它神经网络传入模型前的数据部分都统一了
                """
                if self.is_mlp:
                    tokens_ids_len = len(hotspring_example.tokens_ids)
                    if tokens_ids_len < self.needed_by_mlp_max_seq_len:
                        tokens_ids_len_need = self.needed_by_mlp_max_seq_len - tokens_ids_len
                        for i in range(tokens_ids_len_need):
                            hotspring_example.tokens_ids.append(stoi['PAD'])

                # hotspring_example（object，一个评论），添加到examples（list，n个评论）
                examples.append(hotspring_example)

            # 4、根据词汇表、词向量，构建词嵌入实例

            # new一个词嵌入对象，参数stoi、itos、vectors
            word_embedding = WordEmbedding(stoi, itos)
            word_embedding.vectors = np.array(vectors)

            # 5、数据集tokens_ids、词嵌入两种字典+向量，保存成pkl文件，方便加载
            """
               对于_examples.pkl，mlp和其它神经网络是不一样的；
               对于_word_embedding.pkl，mlp和其它神经网络应该是一样的，
               但是在测试集中，不知道为什么，
               用mlp的原始_word_embedding.pkl，测试其它神经网络，auc一样，
               用其它神经网络的原始_word_embedding.pkl，测试mlp络，auc偏低0.001量级一左右，
               再说，先用着分别保存
            """
            if not self.is_mlp:
                with open(os.path.join(self.data_dir, file_name + '_examples.pkl'), 'wb') as f:
                    pickle.dump(examples, f)
                with open(os.path.join(self.data_dir, file_name + '_word_embedding.pkl'), 'wb') as f:
                    pickle.dump(word_embedding, f)
            else:
                with open(os.path.join(self.data_dir, file_name + '_examples_mlp.pkl'), 'wb') as f:
                    pickle.dump(examples, f)
                with open(os.path.join(self.data_dir, file_name + '_word_embedding_mlp.pkl'), 'wb') as f:
                    pickle.dump(word_embedding, f)
            # 额外保存一下原始词典，较为方便，其实word_embedding对象里面有
            with open("data/dictionary/word2idx.json", 'w+', encoding='utf-8') as f:
                f.write(json.dumps(stoi, ensure_ascii=False))

        # 数据集pkl，存在
        else:
            if not self.is_mlp:
                # 读取数据集pkl
                with open(os.path.join(self.data_dir, file_name + '_examples.pkl'), 'rb') as f:
                    examples = pickle.load(f)
                # 读取词向量pkl
                with open(os.path.join(self.data_dir, file_name + '_word_embedding.pkl'), 'rb') as f:
                    word_embedding = pickle.load(f)
            else:
                with open(os.path.join(self.data_dir, file_name + '_examples_mlp.pkl'), 'rb') as f:
                    examples = pickle.load(f)
                # 读取词向量pkl
                with open(os.path.join(self.data_dir, file_name + '_word_embedding_mlp.pkl'), 'rb') as f:
                    word_embedding = pickle.load(f)

        return examples, word_embedding

    def collate_fn(self, datas):
        """
        记录，这个batch中。
        训练阶段所使用的核对函数，应该就是训练时获取数据的函数。

        :param datas: 一个batch的数据
        :return: 文本，none，文本长度，分类
        """

        # 文本长度list
        seq_lens = []
        # 文本最大长度
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)
        # 文本list
        input_token_ids = []
        # 分类list
        class_label = []

        # 遍历batch（datas）的，每个data
        for data in datas:
            # 当前文本长度
            cur_seq_len = len(data.tokens_ids)
            # add文本长度
            seq_lens.append(len(data.tokens_ids))
            # add文本，padding到相同最大长度
            input_token_ids.append(data.tokens_ids + [self.word_embedding.stoi['PAD']] * (max_seq_len - cur_seq_len))
            # add分类
            class_label.append(data.label)

        # 文本长度转LongTensor
        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        # 文本转LongTensor
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        # 分类转LongTensor
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)

        # 文本，none，文本长度，分类
        return input_token_ids, None, seq_lens, class_label

    def _load_dataset_4_bert(self):
        """
        加载源数据集，bert分词略有不同

        :return: 数据集
        """

        file_name = self.data_name.split('.')[0]

        # 数据集pkl，不存在
        if not os.path.exists(os.path.join(self.data_dir, file_name + '_bert.pkl')):
            examples = []
            raw_data = pd.read_csv(os.path.join(self.data_dir, self.data_name), header=0, names=['label', 'text'])

            for item in tqdm(raw_data.iterrows()):
                hotspring_example = HotspringExample(item[1]['text'], item[1]['label'])
                # bert的tokenizer
                hotspring_example.tokens = self.bert_tokenizer.tokenize(str(hotspring_example.text))
                # bert的encode
                hotspring_example.tokens_ids = self.bert_tokenizer.encode(hotspring_example.tokens,
                                                                          add_special_tokens=True)
                examples.append(hotspring_example)

            # 保存数据集pkl
            with open(os.path.join(self.data_dir, file_name + '_bert.pkl'), 'wb') as f:
                pickle.dump(examples, f)

        # 数据集pkl，存在
        else:
            with open(os.path.join(self.data_dir, file_name + '_bert.pkl'), 'rb') as f:
                examples = pickle.load(f)

        # 数据集
        return examples

    def bert_collate_fn(self, datas):
        """
        记录，这个batch中。
        训练阶段所使用的核对函数，应该就是训练时获取数据的函数。

        :param datas: 一个batch的数据
        :return: 文本，文本遮掩，文本长度，分类
        """

        seq_lens = []
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)
        input_token_ids = []
        class_label = []
        # 文本遮掩
        """
        文本遮掩
        （区别于BERT的MLM）
        （这里的bert_masks，类似于BERT的attention_mask）

        如：[1, 1, 1, 1, 0, 0]
        1代表，要遮掩的部分，即要预测的部分，
        0代表，padding的部分，保持文本同一个长度，

        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        """
        bert_masks = []

        for data in datas:
            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.bert_tokenizer.pad_token_id] * (max_seq_len - cur_seq_len))
            class_label.append(data.label)
            # 添加文本遮掩
            bert_masks.append([1] * len(data.tokens_ids) + [0] * (max_seq_len - cur_seq_len))

        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)
        # 文本遮掩转LongTensor
        bert_masks = torch.ByteTensor(np.array(bert_masks)).to(self.device)

        # 文本，文本遮掩，文本长度，分类
        return input_token_ids, bert_masks, seq_lens, class_label


class HotspringDataSet4TestAndInference(NLPDataSet):
    """
    相对于HotspringDataSet，不一样在
        __init__，
        collate_fn_4_inference,
        bert_collate_fn_4_inference，
        _load_dataset
    """

    def __init__(self, data_dir, data_name, device, test_size=0.3, bert_path=None,
                 is_mlp=False, needed_by_mlp_max_seq_len=260,
                 word_embedding_path=None, model_word_embedding_path=None):

        self.test_size = test_size
        self.device = device
        self.data_dir = data_dir
        self.data_name = data_name
        self.word_embedding_path = word_embedding_path
        self.is_mlp = is_mlp
        self.needed_by_mlp_max_seq_len = needed_by_mlp_max_seq_len
        self.model_word_embedding_path = model_word_embedding_path

        if bert_path:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            self.data = self._load_dataset_4_bert()
            self.word_embedding = None

        else:
            self.data, self.word_embedding = self._load_dataset()

        self.data_set = self.data

    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):

        return len(self.data)

    def _load_dataset(self):

        file_name = self.data_name.split('.')[0]

        if not self.is_mlp:
            examples_path = os.path.join(self.data_dir, file_name + '_examples.pkl')
        else:
            examples_path = os.path.join(self.data_dir, file_name + '_examples_mlp.pkl')
        if not os.path.exists(examples_path):

            # 这里加载的pretrained_wordembedding，其实也就是原始语料的corpus_word_embedding.pkl
            pretrained_wordembedding = load_pretrained_wordembedding(self.word_embedding_path)

            # 加载原始词典
            """
                通过保存的字典方式加载：
                    with open("data/dictionary/word2idx.json", "r", encoding="utf-8") as f:
                        stoi = json.load(f)
            """
            with open(os.path.join(self.model_word_embedding_path), 'rb') as f:
                word_embedding_for_stoi = pickle.load(f)
            stoi = word_embedding_for_stoi.stoi

            # 根据stoi，造itos
            """
                {"zero":0, "one":1} --> {0: 'zero', 1: 'one'}
            """
            itos = {k: v for k, v in enumerate(stoi)}

            vectors = []
            vectors.append(pretrained_wordembedding['UNK'])
            vectors.append(pretrained_wordembedding['PAD'])

            examples = []

            raw_data = pd.read_csv(os.path.join(self.data_dir, self.data_name), header=0, names=['label', 'text'])

            for item in tqdm(raw_data.iterrows()):

                hotspring_example = HotspringExample(item[1]['text'], item[1]['label'])

                hotspring_example.tokens = [*jieba.lcut(str(hotspring_example.text))]

                # 使用字向量
                # hotspring_example.tokens = list(hotspring_example.text)

                # 一句话
                for token in hotspring_example.tokens:
                    # 一个字
                    # 如果token在，原始字典stoi
                    if token in stoi:
                        # 构造vectors：根据token，从pretrained_wordembedding，找到对应的一条词向量，放到vectors
                        vectors.append(pretrained_wordembedding[token])
                        # 构造example：根据原始字典，翻译为这句话的tokens_ids
                        hotspring_example.tokens_ids.append(stoi[token])
                    # 不在字典，直接添加未知token的token_id
                    else:
                        hotspring_example.tokens_ids.append(stoi['UNK'])

                if self.is_mlp:
                    tokens_ids_len = len(hotspring_example.tokens_ids)
                    if tokens_ids_len < self.needed_by_mlp_max_seq_len:
                        tokens_ids_len_need = self.needed_by_mlp_max_seq_len - tokens_ids_len
                        for i in range(tokens_ids_len_need):
                            hotspring_example.tokens_ids.append(stoi['PAD'])
                    else:
                        hotspring_example.tokens_ids = hotspring_example.tokens_ids[:self.needed_by_mlp_max_seq_len]

                examples.append(hotspring_example)

            word_embedding = WordEmbedding(stoi, itos)
            word_embedding.vectors = np.array(vectors)

            if not self.is_mlp:
                with open(os.path.join(self.data_dir, file_name + '_examples.pkl'), 'wb') as f:
                    pickle.dump(examples, f)
                with open(os.path.join(self.data_dir, file_name + '_word_embedding.pkl'), 'wb') as f:
                    pickle.dump(word_embedding, f)
            else:
                with open(os.path.join(self.data_dir, file_name + '_examples_mlp.pkl'), 'wb') as f:
                    pickle.dump(examples, f)
                with open(os.path.join(self.data_dir, file_name + '_word_embedding_mlp.pkl'), 'wb') as f:
                    pickle.dump(word_embedding, f)

        else:
            if not self.is_mlp:
                with open(os.path.join(self.data_dir, file_name + '_examples.pkl'), 'rb') as f:
                    examples = pickle.load(f)
                with open(os.path.join(self.data_dir, file_name + '_word_embedding.pkl'), 'rb') as f:
                    word_embedding = pickle.load(f)
            else:
                with open(os.path.join(self.data_dir, file_name + '_examples_mlp.pkl'), 'rb') as f:
                    examples = pickle.load(f)
                with open(os.path.join(self.data_dir, file_name + '_word_embedding_mlp.pkl'), 'rb') as f:
                    word_embedding = pickle.load(f)
        return examples, word_embedding

    def collate_fn(self, datas):

        seq_lens = []
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)
        input_token_ids = []
        class_label = []

        for data in datas:
            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.word_embedding.stoi['PAD']] * (max_seq_len - cur_seq_len))
            class_label.append(data.label)

        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)

        return input_token_ids, None, seq_lens, class_label

    def collate_fn_4_inference(self, datas):

        seq_lens = []
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)
        input_token_ids = []
        class_label = []

        text_list = []
        for data in datas:
            text_list.append(data.text)

            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.word_embedding.stoi['PAD']] * (max_seq_len - cur_seq_len))
            class_label.append(data.label)

        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)

        # 文本，none，文本长度，分类，原文本
        return input_token_ids, None, seq_lens, class_label, text_list

    def _load_dataset_4_bert(self):

        file_name = self.data_name.split('.')[0]

        if not os.path.exists(os.path.join(self.data_dir, file_name + '_bert.pkl')):
            examples = []
            raw_data = pd.read_csv(os.path.join(self.data_dir, self.data_name), header=0, names=['label', 'text'])

            for item in tqdm(raw_data.iterrows()):
                hotspring_example = HotspringExample(item[1]['text'], item[1]['label'])
                hotspring_example.tokens = self.bert_tokenizer.tokenize(
                    str(hotspring_example.text)[:self.needed_by_mlp_max_seq_len])
                hotspring_example.tokens_ids = self.bert_tokenizer.encode(hotspring_example.tokens,
                                                                          add_special_tokens=True)
                examples.append(hotspring_example)

            with open(os.path.join(self.data_dir, file_name + '_bert.pkl'), 'wb') as f:
                pickle.dump(examples, f)

        else:
            with open(os.path.join(self.data_dir, file_name + '_bert.pkl'), 'rb') as f:
                examples = pickle.load(f)

        return examples

    def bert_collate_fn(self, datas):

        seq_lens = []
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)
        input_token_ids = []
        class_label = []
        bert_masks = []

        for data in datas:
            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.bert_tokenizer.pad_token_id] * (max_seq_len - cur_seq_len))
            class_label.append(data.label)
            bert_masks.append([1] * len(data.tokens_ids) + [0] * (max_seq_len - cur_seq_len))

        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)
        bert_masks = torch.ByteTensor(np.array(bert_masks)).to(self.device)

        return input_token_ids, bert_masks, seq_lens, class_label

    def bert_collate_fn_4_inference(self, datas):

        seq_lens = []
        max_seq_len = len(max(datas, key=lambda x: len(x.tokens_ids)).tokens_ids)
        input_token_ids = []
        class_label = []
        bert_masks = []

        text_list = []
        for data in datas:
            text_list.append(data.text)

            cur_seq_len = len(data.tokens_ids)
            seq_lens.append(len(data.tokens_ids))
            input_token_ids.append(data.tokens_ids + [self.bert_tokenizer.pad_token_id] * (max_seq_len - cur_seq_len))
            class_label.append(data.label)
            bert_masks.append([1] * len(data.tokens_ids) + [0] * (max_seq_len - cur_seq_len))

        seq_lens = torch.LongTensor(np.array(seq_lens)).to(self.device)
        input_token_ids = torch.LongTensor(np.array(input_token_ids)).to(self.device)
        class_label = torch.FloatTensor(np.array(class_label)).to(self.device)
        bert_masks = torch.ByteTensor(np.array(bert_masks)).to(self.device)

        # 文本，文本遮掩，文本长度，分类，文本字符串
        return input_token_ids, bert_masks, seq_lens, class_label, text_list
