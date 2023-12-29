import argparse
import datetime
import random
import torch
import os
import numpy as np
from dataset.concat_data import process_data
from model.mtl.mmoe_with_auxiliary import MMOEAUD
from model.mtl.esmm import ESMM
from model.mtl.mmoe import MMOE
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs')

torch.autograd.set_detect_anomaly(True) # 启动异常检测
# from model.cf.vae import VAECF
# from model.cf.item2vec import Item2Vec

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task_name', default='')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--target_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--sample', type=str, default='random')
    parser.add_argument('--negsample_savefolder', type=str, default='./data/neg_data/')
    parser.add_argument('--negsample_size', type=int, default=99)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--item_min', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')

    parser.add_argument('--model_name', default='aud')
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.0005)

    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--is_parallel', type=bool, default=False)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='l2 regularization') #0.008
    parser.add_argument('--decay_step', type=int, default=5, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for StepLR')
    parser.add_argument('--num_users', type=int, default=1, help='Number of total users')
    parser.add_argument('--num_items', type=int, default=1, help='Number of total items')
    parser.add_argument('--num_embedding', type=int, default=1, help='Number of total source items')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of total labels')
    parser.add_argument('--k', type=int, default=20, help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)')
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 20], help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

    #model param
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden vectors (model)')
    parser.add_argument('--block_num', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_groups', type=int, default=4, help='Number of transformer groups')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for multi-attention')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_mask_prob', type=float, default=0.3,
                        help='Probability for masking items in the training sequence')
    parser.add_argument('--factor_num', type=int, default=128)
    #mtl
    parser.add_argument('--mtl_task_num', type=int, default=2, help='0:like, 1:click, 2:two tasks')
    args = parser.parse_args()
    if args.is_parallel:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device)
    set_seed(args.seed)
    writer = SummaryWriter()
    data_path = './Data Processing/data/QK_article_1w_final.csv'

    train_dataloader, val_dataloader, test_dataloader, categorical_feature_dict, continuous_feature_dict, user_features, labels, le = process_data(args)
    if args.model_name == 'aud':
        model = MMOEAUD(categorical_feature_dict, continuous_feature_dict, user_features, labels, writer,
                        emb_dim=args.embedding_size, device=args.device)
    else:
        model = MMOE(categorical_feature_dict, continuous_feature_dict, labels, writer, emb_dim=args.embedding_size,
                     device=args.device)
    # writer.add_graph(model, next(train_dataloader))
    # print(model)
    model.fit(model, train_dataloader, val_dataloader, test_dataloader, args, le, train=False)
    writer.close()