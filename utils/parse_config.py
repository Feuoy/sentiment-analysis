import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger.logger import setup_logging
from utils.util import read_json, write_json


class ConfigParser:
    """
    写的配置解析器（json和shell）
    """

    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        解析json配置文件，
        处理超参数，for训练、初始化模型、检查点模型、日志模块
        class to parse configuration json file.
        Handles hyperparameters for training, initializations of modules, checkpoint saving and logging module.

        :param config: 配置的json（字典）
        Dict containing configurations, hyperparameters for training. contents of `rnn_config.json` file for example.
        :param resume: 重启checkpoint的路径
        String, path to the checkpoint being loaded.
        :param modification: 修改项（字典）
        Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: 训练唯一标识（用来保存检查点训练日志）
        Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """

        # 训练唯一标识
        if run_id is None:
            # 若run_id为none，用时间戳作为默认run_id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        # 加载配置文件，并更新配置（这个不能乱放，要先放前面）
        self._config = _update_config(config, modification)

        # 项目名
        exper_name = self.config['name']

        # 路径：重启checkpoint
        self.resume = resume

        # 保存路径：模型、模型记录、日志：
        save_dir = Path(self.config['trainer']['saved'])

        # 路径文件名：模型、模型记录
        # /，应该是文件名可以这样连，字符串不行
        self._save_dir = save_dir / 'models' / run_id
        # 路径文件名：日志
        self._log_dir = save_dir / 'log' / run_id

        # 创建目录：模型、模型记录、日志
        """
        run_id为''，exist_ok为True
        parents，父目录不存在，也要创建目录
        exist_ok，？

        pathlib.Path('/my/directory').mkdir(parents=True, exist_ok=True)
        parents：如果父目录不存在，是否创建父目录。
        exist_ok：只有在目录不存在时，创建目录；目录已存在时，不会抛出异常。
        """
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # 写配置：训练配置（√）；模型、模型记录（×）
        # 注意，这里要改，是写死的！
        # 记下run_id的训练配置
        """
        写，这个run_id使用的配置，到（模型/记录）目录
        save updated config file to the checkpoint dir
        """
        write_json(self.config, self.save_dir / 'config.json')

        # 写配置：训练日志
        # 建立日志
        setup_logging(self.log_dir)
        # 日志等级
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def __getitem__(self, name):
        """
        获取self.config的，某一个k的子树
        Access items like ordinary dict.

        :param name:
        :return:
        """

        return self.config[name]

    @classmethod
    def from_args(cls, args, options=''):
        """
        用命令行参数，初始化这个类（训练、测试时）
        Initialize this class from some cli arguments. Used in train, test.
        """

        # 处理options(value)到args(key)

        # 遍历options
        for opt in options:
            # opt，添加到args
            # opt.flags，为单个命令list
            args.add_argument(*opt.flags, default=None, type=opt.type)

        # 若args不是元组
        if not isinstance(args, tuple):
            # parse_args()，转元组
            args = args.parse_args()

        # 用args初始化

        # 初始化计算代理
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        # 初始化config配置文件（重启方式、原始配置方式、命令更新方式）

        # 重启方式（args.resume存在）
        if args.resume is not None:
            # Path
            resume = Path(args.resume)
            # 配置文件路径为：重启点的配置文件
            # 'config.json'
            cfg_fname = resume.parent / 'config.json'
        else:
            # msg：'配置文件需要被指定，请添加'-c rnn_config.json'的命令
            msg_no_cfg = "Configuration file need to be specified. Add '-c rnn_config.json', for example."
            # 保证args.config存在
            assert args.config is not None, msg_no_cfg
            # resume设为none
            resume = None
            # 配置文件路径为：原始配置文件
            cfg_fname = Path(args.config)

        # 读取，配置文件路径（cfg_fname），为json.load
        config = read_json(cfg_fname)
        # 计算代理
        config['device_id'] = args.device
        # 配置文件
        config['config_file_name'] = args.config

        # 命令更新方式（若args.config和resume都存在）（可能这里不是这个意思）
        if args.config and resume:
            # 更新新的配置，来fine-tuning
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # 解析命令行参数进字典
        """
        parse custom cli options into dictionary
        1、遍历options，为opt，
        2、获取opt.flags的所有参数名(key)；从args(keys)中，寻找对应，
        3、opt.target(k): opt.flag(v)
        """
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }

        # 返回这个类对象本身，
        # 可能是保持这个对象，后面继续执行解析新的命令行
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        初始化config内部的模块实例，
        通过给定的name（"dataset"），根据其type（"MLP"），找到一个对应的函数handle的，并给一些合适的参数（"args"/kwargs），初始化其实例模块来返回

        {
            "model_arch": {
                "type": "MLP",
                "args": {...},
                },

            "dataset": {
                "type": "WeiboDataset",
                "args": {...},
            }
        }

        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """

        # 模块
        module_name = self[name]['type']
        # 参数
        module_args = dict(self[name]['args'])

        # 确保kwargs符合module_args
        """
        如果全部kwargs在module_args中，正常
        all()，Return True if bool(x) is True for all values x in the iterable.
        'kwargs存在不在config文件的参数'
        """
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'

        # 根据kwargs，更新module_args
        # update()，D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        module_args.update(kwargs)

        # 用(*args/**module_args)做参数，初始化module中拿到的module_name模块实例，并返回
        """
        getattr()，从module中获取module_name对象
        可以用(, *args, **kwargs)扩展其更多功能，应该是python常用操作
        """
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        与上面的init_obj相比，这个def是，返回function，并可以用偏函数补充
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """

        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)

        # partial(func, *args, **keywords)：偏函数，可以用*args, **keywords扩展func
        return partial(getattr(module, module_name), *args, **module_args)

    def get_logger(self, name, verbosity=2):
        """
        获取logger

        :param name: 日志名（'trainer/train/etc'）
        :param verbosity: 控制台日志显示等级
        :return: logger
        """

        # msg：选择的verbosity无效，可选日志等级为self.log_levels.keys()
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        # 确保verbosity在self.log_levels中，不在的话打印msg_verbosity
        assert verbosity in self.log_levels, msg_verbosity
        # 获取logger
        logger = logging.getLogger(name)
        # 设置日志等级（和显示等级）
        logger.setLevel(self.log_levels[verbosity])
        # logger
        return logger

    @property
    def config(self):
        # @property，把一个方法变成属性调用的，实现get/set
        # 此时配置
        return self._config

    @property
    def save_dir(self):
        # 模型和模型记录路径
        return self._save_dir

    @property
    def log_dir(self):
        # 日志路径
        return self._log_dir


def _get_opt_name(flags):
    """
    解析shell命令（一个一个return）

    :param flags: 输入的命令
    :return:
    """

    # 遍历flags，去掉每个的--开头，去掉一个return一个
    for flag in flags:
        if flag.startswith('--'):
            return flag.replace('--', '')

    # pass
    return flags[0].replace('--', '')


def _update_config(config, modification):
    """
    通过shell更新配置字典，
    helper functions to update config dict with custom cli options

    :param config: 配置
    :param modification: 修改项
    :return: 更新后的config
    """

    # 若modification为none，不修改
    if modification is None:
        return config

    # 遍历modification.items()，更新所有项
    for k, v in modification.items():
        # 若v有值
        if v is not None:
            # 根据{k, v}，更新config，一个更新过程会有一个len(list)减1
            _set_by_path(config, k, v)

    # 更新后的config
    return config


def _set_by_path(tree, keys, value):
    """
    给（config：json），设置keys列表，最后一个k，的v（只更新最后一项）
    （为了更新配置，要设置新配置时使用）
    Set a value in a nested object in tree by sequence of keys.

    :param tree: config
    :param keys: keys列表
    :param value: 新v
    :return:
    """

    # str转list
    """
    所以，注意。
    应该是全部项都有;结尾的

    keys = "a;b;s"，打印['a', 'b', 's']
    keys = "a;b;s;"，打印['a', 'b', 's', '']
    """
    keys = keys.split(';')

    # 获取（config：json）下，所有（keys-1）项（list）（最后一项是分割的''空白）的（k：v）
    # 再取最后一项的（k：v）
    # 赋值新value
    # 只更新最后一项
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """
    访问tree（config：json）结构的，所有keys项
    （为了设置新配置，要访问旧配置时使用）
    Access a nested object in tree by sequence of keys.

    :param tree: （config：json）
    :param keys: keys项
    :return: （json的k:v）
    """

    """
    getitem()，根据keys从tree（config：json）中获取k:v

    reduce(function, sequence[, initial]) -> value
    Apply a function of two arguments cumulatively to the items of a sequence,
    from left to right, so as to reduce the sequence to a single value.
    For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5). 
    If initial is present, it is placed before the items of the sequence in the calculation,
    and serves as a default when the sequence is empty.
    """
    return reduce(getitem, keys, tree)
