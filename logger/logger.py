import logging
import logging.config
from pathlib import Path
from utils.util import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    设置日志配置
    Setup logging configuration

    :param save_dir: 日志保存目录
    :param log_config: 日志配置文件路径
    :param default_level: 默认日志等级
    :return:
    """

    # 解析配置文件路径
    log_config = Path(log_config)

    # 是个文件
    if log_config.is_file():
        # json.load解析
        config = read_json(log_config)

        # 找到并设置日志保存路径
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                # 保存路径，文件名
                handler['filename'] = str(save_dir / handler['filename'])

        # config给logger
        logging.config.dictConfig(config)

    # 没找到
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        # 用默认配置，并info等级
        logging.basicConfig(level=default_level)
