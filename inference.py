import argparse
import collections
import torch
import numpy as np
from utils.util import check_input, write_csv, delete_temp_file
from data_process import data_process as module_data_process
from torch.utils.data import dataloader as module_dataloader
import model.model as module_arch
from utils.parse_config import ConfigParser

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, use_transformers, text_list):
    write_csv(check_input(text_list))

    logger = config.get_logger('inference')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # 预测语料
    inference_dataset = config.init_obj('inference_dataset', module_data_process, device=device)

    if use_transformers:
        inference_dataloader = config.init_obj('inference_data_loader', module_dataloader,
                                               dataset=inference_dataset.data_set,
                                               collate_fn=inference_dataset.bert_collate_fn_4_inference)

        model = config.init_obj('model_arch', module_arch, word_embedding=None)

    else:
        dataset = config.init_obj('dataset', module_data_process, device=device)

        inference_dataloader = config.init_obj('inference_data_loader', module_dataloader,
                                               dataset=inference_dataset.data_set,
                                               collate_fn=inference_dataset.collate_fn_4_inference)

        model = config.init_obj('model_arch', module_arch, word_embedding=dataset.word_embedding)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()

    with torch.no_grad():
        for i, batch_data in enumerate(inference_dataloader):
            input_token_ids, _, seq_lens, class_labels, texts = batch_data

            output = torch.sigmoid(model(input_token_ids, _, seq_lens).squeeze(1)).cpu().detach().numpy()
            class_labels = class_labels.cpu().detach().numpy()

            for text, pred, label in zip(texts, output, class_labels):
                print('----' * 40)

                print(text)

                # 这里直接改分类阈值
                if pred >= 0.5:
                    print("正向情感, 输出值{:.2f}".format(pred))
                else:
                    print("负向情感, 输出值{:.2f}".format(pred))

                if int(label) != 2:
                    tip = "right" if label == round(pred) else "fasle"
                    print(tip)

    delete_temp_file(file_path="data/prediction/text_list_examples_mlp.pkl")
    delete_temp_file(file_path="data/prediction/text_list_word_embedding_mlp.pkl")
    delete_temp_file(file_path="data/prediction/text_list_examples.pkl")
    delete_temp_file(file_path="data/prediction/text_list_word_embedding.pkl")
    delete_temp_file(file_path="data/prediction/text_list_bert.pkl")


def run(config_file, model_path, text_list):
    args = argparse.ArgumentParser(description='text classification')

    # 配置文件、模型路径、计算代理
    args.add_argument('-c', '--config', default=config_file, type=str, help='config file path (default: None)')
    # default=model_path
    args.add_argument('-r', '--resume', default=model_path, type=str, help='path to latest checkpoint (default: None)')
    # default='0',
    args.add_argument('-d', '--device', default='0', type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    print(config.config['model_arch']['type'].lower())

    if 'bert' in config.config['model_arch']['type'].lower():
        main(config, use_transformers=True, text_list=text_list)
    else:
        main(config, use_transformers=False, text_list=text_list)


if __name__ == '__main__':
    text_list = [
        "温泉票住店小孩80，大人90，别墅不清楚，普通的就单次，自助午餐有些坑，大人68元/位，小孩48无/位，没啥东西。",
        "特别好，酒店儿童乐园服务人员魏和平很亲切",
        "早餐种类比较少，牛肉面不错；亲子房很舒服，有四张餐券",
        "The location is wonderful to relax and service staff are very friendly. At the reception desk， they have organised quickly a very nice person from hotel management team to help us， who spoke English and supported us during whole stay in hotel. Thank you Shasha! Also two other nice persons could speak English， which made our stay easy. We had a villa hot spring， which I would recommend， if you are ok not being connected to the main building but having a silent area at the lake with own hot spring bathrobe and wonderful view. To make the hotel more attractive for foreigner， would be good to have some more western food for breakfast and a better coffee :-)",
        "垃圾景区，一生黑，以后不会再去。温泉没特色，要毛巾没毛巾，休息区座位严重不足。庙会什么都要排队，没有一样都系是值钱的，排一次队十分钟，你说三文鱼排队拿还可以接受。最经典的要个炸云吞，现场边包边炸，保证新鲜吗？！！！效率真低，垃圾的不行，花了400大洋买各种不舒服。",
        "到那之后一看到处在搞装修，建房子，乌烟瘴气的。里面也比较破旧，也比较小型。不是都是温泉水来着，能看到是用自来水搀的。性价比低！",
        "拿票要去大门口拿票，价格也不够实惠",
        "好多池都裝修緊，變咗溫泉區活動範圍好細，一陣間就玩完 而且佢嘅池比較細 感覺好迫夾??",
        "好开心啊，景色挺美，非常值得",
        "今天真难过",
    ]

    run('configs/textcnn/textcnn_1d_config.json',
        'saved/text_cnn_1d/models/0705_205906/checkpoint-epoch2.pth',
        text_list)
