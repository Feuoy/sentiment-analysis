import argparse
import collections
import math
import torch
import numpy as np
import model.metric as module_metric
from utils.util import MetricTracker
from data_process import data_process as module_data_process
from torch.utils.data import dataloader as module_dataloader
import model.model as module_arch
from utils.parse_config import ConfigParser

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, use_transformers):
    logger = config.get_logger('test')

    # 1
    # device = torch.device('cpu')
    # 2
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # 测试集语料
    test_dataset = config.init_obj('test_dataset', module_data_process, device=device)

    if use_transformers:
        test_dataloader = config.init_obj('test_data_loader', module_dataloader,
                                          dataset=test_dataset.data_set,
                                          collate_fn=test_dataset.bert_collate_fn_4_inference)

        model = config.init_obj('model_arch', module_arch, word_embedding=None)

    else:
        # 原始语料，只需要dataset，不需要dataloader，拿到dataset.word_embedding，普通神经网网络才需要
        dataset = config.init_obj('dataset', module_data_process, device=device)

        test_dataloader = config.init_obj('test_data_loader', module_dataloader,
                                          dataset=test_dataset.data_set,
                                          collate_fn=test_dataset.collate_fn_4_inference)

        model = config.init_obj('model_arch', module_arch, word_embedding=dataset.word_embedding)

    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(pathlib2.PureWindowsPath(config.resume))
    # checkpoint = torch.load(config.resume.replace('\\', '/'))
    # checkpoint = torch.load("\\saved\\text_cnn_1d\\models\\0706_122111\\checkpoint-epoch15.pth")
    # checkpoint = torch.load(pathlib2.PureWindowsPath(str(config.resume)))
    # checkpoint = torch.load(pathlib.PurePath(config.resume))
    # checkpoint = torch.load(pathlib.PureWindowsPath(config.resume))
    # checkpoint = torch.load(str(pathlib.PureWindowsPath(config.resume)))
    # checkpoint = torch.load(pathlib.PureWindowsPath(os.path.join(str(config.resume))))
    # checkpoint = torch.load(os.path.join(str(config.resume)))
    # checkpoint = torch.load(open(os.path.join(str(config.resume)), 'rb'))
    # checkpoint = torch.load(open(pathlib.joinpath(str(config.resume)), 'rb'))
    checkpoint = torch.load(config.resume)

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # 2
    model = model.cuda()
    model.eval()

    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]
    test_metrics = MetricTracker(*[m.__name__ for m in metric_ftns])

    with torch.no_grad():
        for i, batch_data in enumerate(test_dataloader):
            # 一个batch，128条评论
            input_token_ids, _, seq_lens, class_labels, texts = batch_data

            # 输出值
            output = model(input_token_ids, _, seq_lens).squeeze(1)
            # 真实类别
            class_labels = class_labels

            # bert时候，到时候再写个布尔吧，这样不再多做一点处理（6222%128=78个结尾不去算了）
            if (i + 1) % 8 == 1:
                output_one = output.clone()
                class_labels_one = class_labels.clone()
            elif (i + 1) % 8 == 2:
                output_two = output.clone()
                class_labels_two = class_labels.clone()
            elif (i + 1) % 8 == 3:
                output_three = output.clone()
                class_labels_three = class_labels.clone()
            elif (i + 1) % 8 == 4:
                output_four = output.clone()
                class_labels_four = class_labels.clone()
            elif (i + 1) % 8 == 5:
                output_five = output.clone()
                class_labels_five = class_labels.clone()
            elif (i + 1) % 8 == 6:
                output_six = output.clone()
                class_labels_six = class_labels.clone()
            elif (i + 1) % 8 == 7:
                output_seven = output.clone()
                class_labels_seven = class_labels.clone()
            else:
                pred_tensor = torch.cat((output_one, output_two, output_three, output_four,
                                         output_five, output_six, output_seven, output), 0)
                label_tensor = torch.cat((class_labels_one, class_labels_two, class_labels_three, class_labels_four,
                                          class_labels_five, class_labels_six, class_labels_seven, class_labels), 0)
                for met in metric_ftns:
                    test_metrics.update(met.__name__, met(pred_tensor, label_tensor))

            # # 普通时候
            # for met in metric_ftns:
            #     test_metrics.update(met.__name__, met(output, class_labels))

    test_log = test_metrics.result()

    for k, v in test_log.items():
        logger.info('    {:25s}: {}'.format(str(k), v))

    print(test_log['binary_auc'])

    return test_log['binary_auc']


def run(config_file, model_path):
    args = argparse.ArgumentParser(description='text classification')

    args.add_argument('-c', '--config', default=config_file, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=model_path, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    print(config.config['model_arch']['type'].lower())

    if 'bert' in config.config['model_arch']['type'].lower():
        auc = main(config, use_transformers=True)
    else:
        auc = main(config, use_transformers=False)

    return auc


def find_best_model(config_path, model_path_pre, model_count):
    """
    测试多个模型文件，找测试集的auc也是最高的

    """

    auc_list = []
    for i in range(1, model_count + 1):
        print("----" * 10 + str(i) + "----" * 10)
        auc = run(config_path, model_path_pre + str(i) + '.pth')
        auc_list.append(auc)

    for idx, val in enumerate(auc_list):
        if val == max(auc_list):
            print("the best model file path : {}".format(idx + 1))


def main_threshold(config, use_transformers, threshold):
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')
    test_dataset = config.init_obj('test_dataset', module_data_process, device=device)
    if use_transformers:
        test_dataloader = config.init_obj('test_data_loader', module_dataloader,
                                          dataset=test_dataset.data_set,
                                          collate_fn=test_dataset.bert_collate_fn_4_inference)
        model = config.init_obj('model_arch', module_arch, word_embedding=None)
    else:
        dataset = config.init_obj('dataset', module_data_process, device=device)
        test_dataloader = config.init_obj('test_data_loader', module_dataloader,
                                          dataset=test_dataset.data_set,
                                          collate_fn=test_dataset.collate_fn_4_inference)
        model = config.init_obj('model_arch', module_arch, word_embedding=dataset.word_embedding)
    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    acc_sum = 0
    p_sum = 0
    r_sum = 0
    f1_sum = 0
    auc_sum = 0
    bert_batch_num = 0
    with torch.no_grad():
        for i, batch_data in enumerate(test_dataloader):
            input_token_ids, _, seq_lens, class_labels, texts = batch_data

            output = model(input_token_ids, _, seq_lens).squeeze(1)
            class_labels = class_labels

            # # bert时候
            if (i + 1) % 8 == 1:
                output_one = output.clone()
                class_labels_one = class_labels.clone()
            elif (i + 1) % 8 == 2:
                output_two = output.clone()
                class_labels_two = class_labels.clone()
            elif (i + 1) % 8 == 3:
                output_three = output.clone()
                class_labels_three = class_labels.clone()
            elif (i + 1) % 8 == 4:
                output_four = output.clone()
                class_labels_four = class_labels.clone()
            elif (i + 1) % 8 == 5:
                output_five = output.clone()
                class_labels_five = class_labels.clone()
            elif (i + 1) % 8 == 6:
                output_six = output.clone()
                class_labels_six = class_labels.clone()
            elif (i + 1) % 8 == 7:
                output_seven = output.clone()
                class_labels_seven = class_labels.clone()
            else:
                pred_tensor = torch.cat((output_one, output_two, output_three, output_four,
                                         output_five, output_six, output_seven, output), 0)
                label_tensor = torch.cat((class_labels_one, class_labels_two, class_labels_three, class_labels_four,
                                          class_labels_five, class_labels_six, class_labels_seven, class_labels), 0)
                acc_ok = module_metric.binary_accuracy_threshold(output, class_labels, threshold)
                p_ok = module_metric.binary_precision_threshold(output, class_labels, threshold)
                r_ok = module_metric.binary_recall_threshold(output, class_labels, threshold)
                f1_ok = module_metric.binary_f1_threshold(output, class_labels, threshold)
                auc_ok = module_metric.binary_auc_threshold(pred_tensor, label_tensor, threshold)
                acc_sum += acc_ok
                p_sum += p_ok
                r_sum += r_ok
                f1_sum += f1_ok
                auc_sum += auc_ok
                bert_batch_num += 1
                print(bert_batch_num)
        acc_avg = acc_sum / bert_batch_num
        p_avg = p_sum / bert_batch_num
        r_avg = r_sum / bert_batch_num
        f1_avg = f1_sum / bert_batch_num
        auc_avg = auc_sum / bert_batch_num

    # #         # 普通时候
    #             acc_ok = module_metric.binary_accuracy_threshold(output, class_labels, threshold)
    #             p_ok = module_metric.binary_precision_threshold(output, class_labels, threshold)
    #             r_ok = module_metric.binary_recall_threshold(output, class_labels, threshold)
    #             f1_ok = module_metric.binary_f1_threshold(output, class_labels, threshold)
    #             auc_ok = module_metric.binary_auc_threshold(output, class_labels, threshold)
    #             acc_sum += acc_ok
    #             p_sum += p_ok
    #             r_sum += r_ok
    #             f1_sum += f1_ok
    #             auc_sum += auc_ok
    #             batch_num = i
    #     acc_avg = acc_sum / (batch_num + 1)
    #     p_avg = p_sum / (batch_num + 1)
    #     r_avg = r_sum / (batch_num + 1)
    #     f1_avg = f1_sum / (batch_num + 1)
    #     auc_avg = auc_sum / (batch_num + 1)

    # 打印
    print("----")
    print("acc_avg:", acc_avg)
    print("p_avg:", p_avg)
    print("r_avg:", r_avg)
    print("f1_avg:", f1_avg)
    print("auc_avg:", auc_avg)

    return auc_avg


def run_threshold(config_file, model_path, threshold):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=model_path, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str, help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    print(config.config['model_arch']['type'].lower())
    if 'bert' in config.config['model_arch']['type'].lower():
        auc = main_threshold(config, use_transformers=True, threshold=threshold)
    else:
        auc = main_threshold(config, use_transformers=False, threshold=threshold)
    return auc


def find_best_threshold(config_file, model_path, threshold_start, threshold_end):
    # list
    threshold_list = [i / 100 for i in range(int(threshold_start * 100), int((threshold_end + 0.01) * 100))]
    auc_list = []

    print("\t\t\t\t\t\tthreshold\t\tauc")

    # loop
    for threshold in threshold_list:
        auc_ = run_threshold(config_file, model_path, threshold)
        auc_list.append(auc_)
        print("\t\t\t\t\t\t{:.2}\t\t\t{}".format(threshold, auc_))

    # find
    for idx, val in enumerate(auc_list):
        if math.isclose(val, max(auc_list), abs_tol=1e-5):
            print("\nthe best:\tthreshold: {},\t\tauc: {}".format(threshold_list[idx], val))


if __name__ == '__main__':
    # 1、测试一个模型文件，
    # run('configs/textcnn/textcnn_1d_config.json', 'saved/text_cnn_1d/models/0706_122111/checkpoint-epoch15.pth')

    # 2、找最佳auc的模型文件，
    # find_best_model('configs/textcnn/textcnn_1d_config.json', 'saved/text_cnn_1d/models/0705_172455/checkpoint-epoch',
    #                 model_count=5)

    # 3、找一个模型文件的最佳分类阈值
    # 4、也可以用作使用阈值测试
    # # =======================================================
    find_best_threshold(
        'configs/mlp/mlp_1h_config.json',
        'saved/mlp_1h/models/0706_140218/checkpoint-epoch14.pth',
        threshold_start=0.96, threshold_end=0.96)
    # # =======================================================
    # find_best_threshold(
    #     'configs/mlp/mlp_2h_config.json',
    #     'saved/mlp_2h/models/0706_135731/checkpoint-epoch15.pth',
    #     threshold_start=0.99, threshold_end=0.99)
    # # =======================================================
    # find_best_threshold(
    #     'configs/mlp/mlp_4h_config.json',
    #     'saved/mlp_4h/models/0706_143131/checkpoint-epoch19.pth',
    #     threshold_start=0.94, threshold_end=0.94)
    # # =======================================================
    # find_best_threshold(
    #     'configs/textcnn/textcnn_1d_config.json',
    #     'saved/text_cnn_1d/models/0706_122111/checkpoint-epoch15.pth',
    #     threshold_start=0.89, threshold_end=0.89)
    # # =======================================================
    # find_best_threshold(
    #     'configs/textcnn/textcnn_2d_config.json',
    #     'saved/text_cnn_2d/models/0706_123642/checkpoint-epoch15.pth',
    #     threshold_start=0.93, threshold_end=0.93)
    # # =======================================================
    # find_best_threshold(
    #     'configs/rnn/rnn_config.json',
    #     'saved/rnn/models/0706_124412/checkpoint-epoch49.pth',
    #     threshold_start=0.89, threshold_end=0.89)
    # # =======================================================
    # find_best_threshold(
    #     'configs/rnn/lstm_config.json',
    #     'saved/lstm/models/0706_151222/checkpoint-epoch31.pth',
    #     threshold_start=0.59, threshold_end=0.59)
    # # =======================================================
    # find_best_threshold(
    #     'configs/rnn/gru_config.json',
    #     'saved/gru/models/0706_144335/checkpoint-epoch15.pth',
    #     threshold_start=0.50, threshold_end=0.50)
    # # # =======================================================
    # find_best_threshold(
    #     'configs/attention/rnn_attention_config.json',
    #     'saved/rnn_attention/models/0709_123209/checkpoint-epoch15.pth',
    #     threshold_start=0.89, threshold_end=0.89)
    # # =======================================================
    # find_best_threshold(
    #     'configs/textrcnn/textrcnn_config.json',
    #     'saved/rcnn/models/0709_130446/checkpoint-epoch12.pth',
    #     threshold_start=0.92, threshold_end=0.92)
    # # # =======================================================
    # find_best_threshold(
    #     'configs/bert/bert_config.json',
    #     'saved/bert/models/0709_021756/checkpoint-epoch4.pth',
    #     threshold_start=0.91, threshold_end=0.91)
    # # =======================================================
    # find_best_threshold(
    #     'configs/bert/bert_cnn_config.json',
    #     'saved/bert_cnn/models/0709_042735/checkpoint-epoch6.pth',
    #     threshold_start=0.97, threshold_end=0.97)
    # # =======================================================
    # find_best_threshold(
    #     'configs/bert/bert_rnn_attention_config.json',
    #     'saved/bert_rnn_attention/models/0709_091857/checkpoint-epoch4.pth',
    #     threshold_start=0.89, threshold_end=0.89)
    # # =======================================================
    # find_best_threshold(
    #     'configs/bert/bert_rcnn_config.json',
    #     'saved/bert_rcnn/models/0709_064734/checkpoint-epoch2.pth',
    #     threshold_start=0.93, threshold_end=0.93)
