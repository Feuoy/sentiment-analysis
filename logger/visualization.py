import importlib
from datetime import datetime


class TensorboardWriter():
    """
    TensorboardWriter，做可视化
    """

    def __init__(self, log_dir, logger, enabled):
        """

        :param log_dir: 日志目录
        :param logger: logger
        :param enabled:
        """

        # 写对象
        self.writer = None
        # 被选择的pytorch可视化工具
        self.selected_module = ""

        # 如果可
        if enabled:
            # 日志目录，转str
            log_dir = str(log_dir)

            # Retrieve vizualization writer.

            # 遍历两个pytorch可视化工具，选用其中一个
            # for module in ["torch.utils.tensorboard", "tensorboardX"]:
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    # 写对象，
                    # SummaryWriter将log_dir读取，module将其转换为self.writer
                    # SummaryWriter，Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    # 选择成功
                    succeeded = True
                    break
                except ImportError:
                    # 选择失败
                    succeeded = False

                # 被选择的pytorch可视化工具，确定
                self.selected_module = module

            # 若没有成功选择一个可视化工具
            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                          "this machine. Please install TensorboardX with 'pip install tensorboardx', " \
                          "upgrade PyTorch to version >= 1.1 to use 'torch.utils.tensorboard' " \
                          " or turn off the option in the 'rnn_config.json' file."
                logger.warning(message)

        # 写对象，的函数列表
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars',
            'add_image', 'add_images',
            'add_audio', 'add_text',
            'add_histogram', 'add_pr_curve',
            'add_embedding'
        }

        # 模式例外list
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}

        # 步数（单位）
        self.step = 0
        # 模式（训练验证等）
        self.mode = ''

        # 计时器，赋值now
        self.timer = datetime.now()

    def __getattr__(self, name):
        """
        返回某个属性可视化对象（某一个函数的装饰器）

        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """

        """
        # with self.writer() as w:
        # w.add_graph(self.model)
        # self.writer.new_graph(self.model)
        # def new_graph(self, model):
        # self.writer.add_graph(model)
        """

        # 若需要的name函数名在定义中
        if name in self.tb_writer_ftns:
            # add_data，从self.writer中获取的name函数
            # getattr()，
            # getattr(object, name[, default]) -> value
            # Get a named attribute from an object;
            add_data = getattr(self.writer, name, None)

            # 做一个装饰器
            def wrapper(tag, data, *args, **kwargs):
                """

                :param tag: 模式标记
                :param data: 数据
                :param args:
                :param kwargs:
                :return:
                """

                # 若拿到了add_data
                if add_data is not None:

                    # 若name不在模式例外list
                    if name not in self.tag_mode_exceptions:
                        # mode(train/valid) tag，模式标记
                        tag = '{}/{}'.format(tag, self.mode)

                    # [模式标记，数据，步数]添加到add_data（从self.writer中，获取的name函数）
                    add_data(tag, data, self.step, *args, **kwargs)

            # 调用
            return wrapper

        # 需要的name函数名不在定义中
        else:
            # 找到默认返回
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                # attr，赋值，"object对象"的name函数名属性
                attr = object.__getattr__(name)
            except AttributeError:
                # pytorch可视化工具（self.selected_module），没有name这个函数名属性
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))

            # 如果找到了的，"object对象"的name函数名属性
            return attr

    def set_step(self, step, mode='train'):
        """
        设置step步数，一个batch_size计算一次时长

        :param step:
        :param mode:
        :return:
        """

        # 模式（训练等）
        self.mode = mode
        # 步数（单位）
        self.step = step

        # step为0
        if step == 0:
            # 重新计时
            self.timer = datetime.now()

        # 否则
        else:
            # 求时长
            duration = datetime.now() - self.timer
            # 添加标量：每秒几个step（1秒/duration秒数）
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            # 重设计时器
            self.timer = datetime.now()
