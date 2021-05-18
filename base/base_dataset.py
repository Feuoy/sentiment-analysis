from torch.utils.data import Dataset


class WordEmbedding():
    """
    词嵌入对象，词向量
    """

    def __init__(self, stoi, itos):
        """
        stoi: 字典，token键，index值；'UNK': 0
        itos: 也是字典，index键，token值；0: 'UNK'
        vectors: 词向量集合
        """

        self.stoi = stoi
        self.itos = itos
        self.vectors = None


class Example():
    """
    数据集对象，语料
    """

    def __init__(self, text, label):
        # 文本
        self.text = text
        # 嵌入，['UNK',]
        self.tokens = []
        # 嵌入index，[0,] ————需要的
        self.tokens_ids = []

        # 分类
        self.label = label
        # 分类id字典，{负/正：0/1}
        self.label_id_map = {}
        # 分类id，根据label获取 ————需要的
        self.label_id = self.label_id_map[label]


class NLPDataSet(Dataset):
    """
    Dataset，数据集构建器
    """

    def __init__(self, data_dir, data_name, batch_first=False, data_split=[0.3, 0.2]):
        """

        :param data_dir:  数据集路径
        :param data_name:  数据集文件名
        :param batch_first: 是否在第一维产生batch维，默认False
        :param data_split:
            验证集、测试集的划分list，
            默认：训练集0.5，验证集0.2，测试集0.3，
            1没用这里的，用的data_process的test_size，
            2被内层封装实现了
        """

        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_first = batch_first
        self.data_split = data_split
        # 加载源数据集
        self.data = self._load_dataset()

    def _load_dataset(self):
        """
        加载源数据集
        """

        raise NotImplementedError

    def __len__(self):
        """
        数据集长度
        """

        raise NotImplementedError

    def __getitem__(self, index):
        """
        获取一个item
        """

        raise NotImplementedError
