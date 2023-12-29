import argparse
import collections

import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm


raw_log_path = './dataset/data/log_standard_4_08_to_4_21_pure.csv'
user_feature_path = './dataset/data/user_features_pure.csv'
video_feature_path = './dataset/data/video_features_basic_pure.csv'
save_path = './dataset/data/save/'
category_features = (['user_id', 'weekday', 'hourmin', 'user_active_degree', 'is_video_author',
                     'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range']
                     + [f'onehot_feat{i}' for i in range(18)]
                     + ['video_id', 'author_id', 'upload_type', 'tag'])
continuous_features = ['duration_ms', 'server_width', 'server_height', 'follow_user_num', 'fans_user_num', 'friend_user_num']
user_features = (['weekday', 'hourmin', 'user_active_degree', 'is_video_author',
                     'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range', 'follow_user_num', 'fans_user_num', 'friend_user_num'] +
                [f'onehot_feat{i}' for i in range(18)])
labels = ['effective_view', 'is_like', 'long_view', 'is_follow', 'is_comment', 'is_forward', 'is_not_hate']
history_length_max_per_user = 20
history_length_min_per_user = 3
history_id_columns = [f'history_id_{i}' for i in range(1, history_length_max_per_user + 1)]
history_tag_columns = [f'history_tag_{i}' for i in range(1, history_length_max_per_user + 1)]
gen_columns = history_tag_columns + history_id_columns + ['emp_' + label for label in labels] + ['flag']

class mtlDataSet(data_utils.Dataset):
    def __init__(self, data):
        self.feature = data[0]
        self.label = data[1]

    def __getitem__(self, index):
        feature = self.feature[index]
        label = torch.FloatTensor(self.label[index])
        return feature, label

    def __len__(self):
        return len(self.feature)

def concat_features(raw_log_path, user_feature_path, video_feature_path):
    raw_df = pd.read_csv(raw_log_path)
    user_df = pd.read_csv(user_feature_path)
    video_df = pd.read_csv(video_feature_path)
    raw_df = raw_df.merge(user_df, on=['user_id'])
    raw_df = raw_df.merge(video_df, on=['video_id'])
    # 处理时间
    raw_df['hourmin'] = raw_df['hourmin'] // 100
    # 加入星期几特征
    raw_df['date'] = pd.to_datetime(raw_df['date'], format='%Y%m%d')
    raw_df['weekday'] = raw_df['date'].dt.day_name()
    # 处理视频时长
    raw_df['duration_ms'] = raw_df['duration_ms'] // 100

    # 处理tag
    raw_df['tag'] = raw_df['tag'].apply(lambda x:str(x).split(',')[0])

    # 处理hate标签
    raw_df['is_not_hate'] = 1 - raw_df['is_hate']

    # 处理离散变量
    le = dict()
    for fea in category_features:
        le[fea] = LabelEncoder()
        raw_df[fea] = le[fea].fit_transform(raw_df[fea])

    # # 处理连续变量
    # num_bins = 11
    # for fea in continuous_features:
    #     # print(raw_df[fea])
    #     raw_df[fea + '_bin'] = pd.qcut(raw_df[fea], q=num_bins, labels=False, duplicates='drop')
    #     n = raw_df[fea + '_bin'].nunique()
    #     raw_df[fea + '_bin'] = raw_df[fea + '_bin'].map(dict((idx, idx / (n - 1)) for idx in range(n)))
    #     # print(pd.qcut(raw_df[fea], q=num_bins, labels=False, duplicates='drop', retbins=True))
    #     # print(raw_df[fea + '_bin'].value_counts())

    # 处理连续变量
    mms = MinMaxScaler(feature_range=(0, 1))
    raw_df[continuous_features] = mms.fit_transform(raw_df[continuous_features])

    # 处理成字典
    final_columns = category_features + continuous_features
    categorical_feature_dict, continuous_feature_dict = {}, {}
    for idx, col in tqdm(enumerate(final_columns)):
        if col in category_features:
            raw_df[col] = raw_df[col].map(int)
            categorical_feature_dict[col] = (len(raw_df[col].unique()), idx)
        else:
            continuous_feature_dict[col] = (0, idx)
    return raw_df, categorical_feature_dict, continuous_feature_dict, le

def add_history_actions(raw_df):
    user_history_id_record = collections.defaultdict(list)
    user_history_tag_record = collections.defaultdict(list)
    emp_xtr_record = dict((label, collections.defaultdict(list)) for label in labels)
    user_item_record = collections.defaultdict(list)
    # 使用NumPy数组进行操作

    history_data = np.zeros((raw_df.shape[0], 2 * history_length_max_per_user + len(labels) + 1), dtype=np.int64)
    raw_df = raw_df.sort_values('time_ms', ascending=True).reset_index(drop=True)
    for i in tqdm(range(raw_df.shape[0])):
        user_id = raw_df.loc[i, 'user_id']
        item_id = raw_df.loc[i, 'video_id']
        tag_id = raw_df.loc[i, 'tag']
        curr_len = len(user_history_id_record[user_id])
        # 填入用户历史行为
        if curr_len >= history_length_max_per_user:
            history_id = user_history_id_record[user_id]
            history_tag = user_history_tag_record[user_id]
        else:
            history_id = [-1] * (history_length_max_per_user - curr_len) + user_history_id_record[user_id]
            history_tag = [-1] * (history_length_max_per_user - curr_len) + user_history_tag_record[user_id]
        # 填入emp_xtr
        xtr_list = []
        n = len(user_item_record[user_id])
        flag = False
        if n == history_length_max_per_user:
            post_item_id = user_item_record[user_id].pop(0)
            flag = True
        for label in labels:
            if n == 0:
                xtr_list.append(0)
            else:
                # print(emp_xtr_record[label][user_id], n)
                xtr_list.append(len(emp_xtr_record[label][user_id]) / n)
            # 只计算20个item内的emp_xtr
            if flag and emp_xtr_record[label][user_id] and (post_item_id == emp_xtr_record[label][user_id][0]):
                emp_xtr_record[label][user_id].pop(0)
            if raw_df.loc[i, label]:
                emp_xtr_record[label][user_id].append(item_id)
        # 确定这条样本是否保留，如果小于历史记录最小长度则去掉
        if curr_len >= history_length_min_per_user:
            history = np.concatenate([history_tag, history_id, xtr_list, [True]])
        else:
            history = np.concatenate([history_tag, history_id, xtr_list, [False]])
        # 使用NumPy数组进行赋值
        # print(history)
        history_data[i] = history
        if raw_df.loc[i, 'effective_view']:
            user_history_id_record[user_id].append(item_id)
            user_history_tag_record[user_id].append(tag_id)
            curr_len += 1
            if curr_len >= history_length_max_per_user:
                user_history_id_record[user_id].pop(0)
                user_history_tag_record[user_id].pop(0)

        user_item_record[user_id].append(item_id)
    raw_df[gen_columns] = history_data
    full_df = raw_df[raw_df['flag'] == 1].reset_index(drop=True).copy()
    del raw_df
    return full_df


def split_train_test_by_time(df):
    df = df.sort_values('time_ms', ascending=True).reset_index(drop=True)
    final_columns = category_features + continuous_features + gen_columns + labels
    df = df[final_columns]
    # print(df.head())
    train_length = int(df.shape[0]*0.8)
    val_length = int(df.shape[0]*0.1)
    train_df = df[:train_length]
    val_df = df[train_length:train_length + val_length]
    test_df = df[train_length + val_length:]
    train_df.to_csv(save_path + 'train_data.csv', index=False)
    val_df.to_csv(save_path + 'val_data.csv', index=False)
    train_df.to_csv(save_path + 'test_data.csv', index=False)
    del df
    # print(train_df.head())
    return train_df, val_df, test_df

def load_data(train_df, val_df, test_df, args):
    train_dataset = (train_df.loc[:, ~train_df.columns.isin(labels)].values, train_df.loc[:, train_df.columns.isin(labels)].values)
    val_dataset = (val_df.loc[:, ~val_df.columns.isin(labels)].values, val_df.loc[:, val_df.columns.isin(labels)].values)
    test_dataset = (test_df.loc[:, ~test_df.columns.isin(labels)].values, test_df.loc[:, test_df.columns.isin(labels)].values)
    train_dataset = mtlDataSet(train_dataset)
    val_dataset = mtlDataSet(val_dataset)
    test_dataset = mtlDataSet(test_dataset)

    # dataloader
    train_dataloader = get_train_loader(train_dataset, args)
    val_dataloader = get_val_loader(val_dataset, args)
    test_dataloader = get_test_loader(test_dataset, args)
    return train_dataloader, val_dataloader, test_dataloader

def get_train_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_val_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_test_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    return dataloader
def process_data(args):
    raw_df, categorical_feature_dict, continuous_feature_dict, le = concat_features(raw_log_path, user_feature_path, video_feature_path)
    full_df = add_history_actions(raw_df)

    train_df, val_df, test_df = split_train_test_by_time(full_df)
    train_dataloader, val_dataloader, test_dataloader = load_data(train_df, val_df, test_df, args)
    return train_dataloader, val_dataloader, test_dataloader, categorical_feature_dict, continuous_feature_dict, user_features, labels, le

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--is_parallel', type=bool, default=False)
    args = parser.parse_args()
    process_data(args)