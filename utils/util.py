import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import pandas as pd
import os
import warnings


def check_input(text_list):
    """
    检查预测的输入list

    :param text_list:
    :return:
    """

    # 一个str的话，转list
    if isinstance(text_list, str):
        text_list = [text_list, ]

    # 处理前，文本个数、最大长度
    len_ = len(text_list)
    max_len_ = max([len(i) for i in text_list])

    # 去长度为0
    text_list = [i for i in text_list if len(i) != 0]
    # 取长度最大256
    for idx, text in enumerate(text_list):
        if len(text) > 256:
            text_list[idx] = text[:256]

    # 提醒
    if len(text_list) == 0:
        raise NotImplementedError("输入的文本全部为空, 长度为0！")
    if len(text_list) < len_:
        warnings.warn("输入的文本中有长度为0的句子, 它们将被忽略掉！")
        # print("输入的文本中有长度为0的句子, 它们将被忽略掉！")
    if max_len_ > 256:
        warnings.warn("输入的文本中有长度大于256的句子, 它们将被截断掉！")
        # print("输入的文本中有长度大于256的句子, 它们将被截断掉！")

    return text_list


def write_csv(text_list):
    """
    将输入的预测list写为csv，会覆盖原来的csv

    :param text_list:
    :return:
    """

    df = pd.DataFrame({'label': {}, 'review': {}})
    for idx, val in enumerate(text_list):
        # 2，代表暂无分类
        new_line = [str(2), val]
        df.loc[idx] = new_line
    df.to_csv("data/prediction/text_list.csv", index=False)


def delete_temp_file(file_path):
    """
    删除预测list产生的，文本嵌入pkl、词向量pkl

    :return:
    """

    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file \'" + file_path + "\' does not exist.")


def ensure_dir(dirname):
    """
    确保目录存在，不存在的话创建目录
    """

    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(filename):
    """
    读取json

    :param filename: 文件路径
    :return: json.loads()
    """

    # Path()，PurePath subclass that can make system calls.
    filename = Path(filename)

    """
    ========= ===============================================================
    Character Meaning
    --------- ---------------------------------------------------------------
    'r'       open for reading (default)
    'w'       open for writing, truncating the file first
    'x'       create a new file and open it for writing
    'a'       open for writing, appending to the end of the file if it exists
    'b'       binary mode
    't'       text mode (default)
    '+'       open a disk file for updating (reading and writing)
    'U'       universal newline mode (deprecated)
    ========= ===============================================================
    """

    # rt模式下，r读取文本时，t会自动把\r\n转换成\n.（文本模式）
    with filename.open('rt') as handle:
        # 返回一个json.loads()，OrderedDict类型的，有序字典？
        # object_hook=，用object_hook这种类对象返回
        """
        json.load()，加载python json格式文件
        json.loads()，解码python json格式
        """
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, filename):
    """
    写一个json（不一定是新的）

    :param content: 内容，json
    :param filename: 文件路径和名
    :return: json.loads()
    """

    filename = Path(filename)

    # wt，“写+文本”模式
    with filename.open('wt') as handle:
        # indent，缩进
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """
    让传进来的data_loader可以一直重复循环里面的data，信息循环
    wrapper function for endless data loader.

    :param data_loader:
    :return: data_loader里面的loader里面的一项
    """

    # repeat()，一直重复返回迭代对象
    for loader in repeat(data_loader):
        # 迭代返回loader里面的data
        yield from loader


class MetricTracker:
    """
    作一个度量工具对象（只用来记录度量结果）
    """

    def __init__(self, *keys, writer=None):

        # 写对象
        self.writer = writer

        # 数据对象（pd.DataFrame），
        # 行：keys，即accuracy、f1、auc
        # 列：[总量, 个数, 平均]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])

        # 清空数据
        self.reset()

    def reset(self):
        """
        清空

        :return:
        """

        # 遍历列
        for col in self._data.columns:
            # 每列的值全赋0
            # 行index不消除的
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
        更新

        :param key: 行
        :param value: 某些列的value单位
        :param n:
        :return:
        """

        # 若self.writer对象存在，指针意思？
        if self.writer is not None:
            # add_scalar()，添加key, value到self.writer
            self.writer.add_scalar(key, value)

        # 总量，n个单位value
        self._data.total[key] += value * n
        # 个数，n个
        self._data.counts[key] += n
        # 平均，总量/个数
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def get_counts_of_key(self, key):
        return self._data.counts[key]

    def avg(self, key):
        """
        获取average列的，指定key行

        :param key:
        :return:
        """
        return self._data.average[key]

    def result(self):
        """
        获取{ key: average }字典
        :return:
        """

        return dict(self._data.average)
