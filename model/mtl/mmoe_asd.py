'''
Reference:
    [1]Jiaqi Ma et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD
    International Conference on Knowledge Discovery & Data Mining, pages 1930–1939, 2018.
Reference:
    https://github.com/busesese/MultiTaskModel
'''
import collections
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from model.layers.inputs import varlen_embedding_lookup, get_varlen_pooling_list
from ..layers.attention import SelfAttentionModule
from ..layers.core import DNN, PredictionLayer
from sklearn.metrics import roc_auc_score
save_data_path = './dataset/data/save/'


class MMOEASD(nn.Module):
    """
    MMOE for CTCVR problem
    """

    def __init__(self, categorical_feature_dict, continuous_feature_dict, var_cat_feature_dict, user_features, labels, writer, emb_dim=128,
                 num_experts=3, expert_dnn_hidden_units=(256, 128),
                 gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,), user_tower_dnn_hidden_units=(256, 128, 64,),
                 l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=True,
                 device='cpu', gpus=None):
        """
        MMOE model input parameters
        :param user_feature_dict: user feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param item_feature_dict: item feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param emb_dim: int embedding dimension
        :param n_expert: int number of experts in mmoe
        :param mmoe_hidden_dim: mmoe layer input dimension
        :param hidden_dim: list task tower hidden dimension
        :param dropouts: list of task dnn drop out probability
        :param output_size: int task output size
        :param expert_activation: activation function like 'relu' or 'sigmoid'
        :param num_task: int default 2 multitask numbers
        """
        super(MMOEASD, self).__init__()
        torch.manual_seed(seed)
        self.regularization_weight = []
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        # check input parameters
        if categorical_feature_dict is None or continuous_feature_dict is None:
            raise Exception("input parameter categorical_feature_dict must be not None")
        if isinstance(categorical_feature_dict, dict) is False or isinstance(continuous_feature_dict, dict) is False:
            raise Exception("input parameter categorical_feature_dict must be dict")
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.user_dnn_hidden_units = user_tower_dnn_hidden_units
        self.categorical_feature_dict = categorical_feature_dict
        self.continuous_feature_dict = continuous_feature_dict
        self.var_cat_feature_dict = var_cat_feature_dict
        self.user_features = user_features
        self.num_tasks = len(labels)
        self.num_experts = num_experts
        self.writer = writer
        self.labels = labels
        self.task_types = ['binary']*self.num_tasks
        self.eps = torch.FloatTensor([1e-8]).to(device)
        # TODO
        self.loss_function = [F.binary_cross_entropy]*self.num_tasks
        # print(len(self.categorical_feature_dict), len(self.continuous_feature_dict), len(self.history_feature_dict))

        if device:
            self.device = device

        # embedding初始化
        self.embedding_dict = nn.ModuleDict(
            {feat: nn.Embedding(num[0], emb_dim, sparse=False) # 非one-hot编码
             for feat, num in
             categorical_feature_dict.items()}
        )

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)

        # user embedding + item embedding
        self.input_dim = emb_dim * (len(self.categorical_feature_dict) + len(self.var_cat_feature_dict)) + len(continuous_feature_dict)
        self.user_cat_feature_len = len(set(self.categorical_feature_dict.keys()) & set(self.user_features))
        self.user_input_dim = emb_dim * (self.user_cat_feature_len + len(self.var_cat_feature_dict)) + 1#  + len(self.user_features) - self.user_cat_feature_len + 1

        # expert dnn
        self.expert_dnn = nn.ModuleList([DNN(self.input_dim, self.expert_dnn_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std, device=device) for _ in range(self.num_experts)])

        # gate dnn
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, gate_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim,
                       self.num_experts, bias=False) for _ in range(self.num_tasks)])

        # tower dnn (task-specific)
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        # 直接默认塔网络有多层
        self.self_attention = SelfAttentionModule(tower_dnn_hidden_units[-1], self.num_tasks)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1,
            bias=False) for _ in range(self.num_tasks)])

        # user tower
        self.user_tower_dnn = nn.ModuleList(
            [DNN(self.user_input_dim, user_tower_dnn_hidden_units, activation=dnn_activation,
                 l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                 init_std=init_std, device=device) for _ in range(self.num_tasks)])
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.user_tower_dnn.named_parameters()),
            l2=l2_reg_dnn)
        self.user_tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            user_tower_dnn_hidden_units[-1] if len(user_tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1,
            bias=False) for _ in range(self.num_tasks)])

        # # item tower
        # self.item_tower_dnn = nn.ModuleList(
        #     [DNN(self.item_input_dim, tower_dnn_hidden_units, activation=dnn_activation,
        #          l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
        #          init_std=init_std, device=device) for _ in range(self.num_tasks)])
        # self.add_regularization_weight(
        #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.item_tower_dnn.named_parameters()),
        #     l2=l2_reg_dnn)
        # self.item_tower_dnn_final_layer = nn.ModuleList([nn.Linear(
        #     tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1,
        #     bias=False) for _ in range(self.num_tasks)])


        regularization_modules = [self.expert_dnn, self.gate_dnn_final_layer, self.tower_dnn_final_layer, self.user_tower_dnn_final_layer]# , self.item_tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

        self.out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])
        self.user_out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])
        # self.item_out = nn.ModuleList([PredictionLayer(task) for task in self.task_types])


    def forward(self, x):
        # assert x.size()[1] == len(self.categorical_feature_dict) + len(self.continuous_feature_dict)
        # embedding
        cat_embed_list, con_embed_list, user_cat_embed_list = list(), list(), list()
        for cat_feature, num in self.categorical_feature_dict.items():
            cat_embed_list.append(self.embedding_dict[cat_feature](x[:, num[1]].long()))
            if cat_feature in self.user_features:
                user_cat_embed = self.embedding_dict[cat_feature](x[:, num[1]].long()).detach()
                user_cat_embed_list.append(user_cat_embed)

        for con_feature, num in self.continuous_feature_dict.items():
            con_embed_list.append(x[:, num[1]].unsqueeze(1))
            # if con_feature in self.user_features:
                # user_con_embed = x[:, num[1]].unsqueeze(1).detach()
                # user_con_embed_list.append(user_con_embed)

        sequence_embed_dict = varlen_embedding_lookup(x, self.embedding_dict, self.var_cat_feature_dict)
        varlen_embed_list = get_varlen_pooling_list(sequence_embed_dict, x, self.var_cat_feature_dict, self.device)

        # embedding concat
        cat_embed = torch.cat(cat_embed_list, axis=1)
        con_embed = torch.cat(con_embed_list, axis=1)
        varelen_embed = torch.cat(varlen_embed_list, axis=1)

        # 用户辅助网络 embedding concat
        user_cat_embed = torch.cat(user_cat_embed_list, axis=1)

        # hidden layer
        dnn_input = torch.cat([cat_embed, con_embed, varelen_embed], axis=1).float()  # batch * hidden_size
        user_dnn_input = torch.cat([user_cat_embed, varelen_embed], axis=1).float()

        # expert dnn
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)
        # print(expert_outs.shape)
        # print(expert_outs)
        # gate dnn
        mmoe_outs = []
        for i in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)  # (bs, 1, dim)
            mmoe_outs.append(gate_mul_expert.squeeze())

        # tower dnn (task-specific)
        task_dnn_outs, user_weights = [], []
        for i, label in enumerate(self.labels):
            tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
            task_dnn_outs.append(tower_dnn_out)
            # 计算当前任务辅助网络输入
            emp_label = x[:, self.continuous_feature_dict['emp_'+label][1]].unsqueeze(1)# .detach()
            curr_user_dnn_input = torch.cat([user_dnn_input, emp_label], axis=1).float()
            user_tower_dnn_out = self.user_tower_dnn[i](curr_user_dnn_input)
            # item_tower_dnn_out = self.item_tower_dnn[i](item_dnn_input)
            user_tower_dnn_logit = self.user_tower_dnn_final_layer[i](user_tower_dnn_out)
            # user辅助塔结果处理
            user_output = self.user_out[i](user_tower_dnn_logit)
            user_weights.append(user_output)

        attention_input = torch.stack(task_dnn_outs, dim=1)
        attention_output = self.self_attention(attention_input)
        task_final_layer_input = torch.split(attention_output + attention_input, 1, dim=1)
        task_outs = []
        for i, label in enumerate(self.labels):
            tower_dnn_logit = self.tower_dnn_final_layer[i](task_final_layer_input[i].squeeze(1))
            # 主塔结果处理
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)

        task_outs = torch.cat(task_outs, -1)
        user_weights = torch.cat(user_weights, -1) # (batch size, num_tasks)
        # item_weight = torch.cat(item_weights, -1)
        return task_outs, user_weights# , item_weight
    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def fit(self, model, train_loader, val_loader, test_loader, args, le, train=True):
        device = args.device
        epoch = args.epochs
        early_stop = 5
        path = os.path.join(args.save_path,
                            '{}_{}_seed{}_best_model_{}.pth'.format(args.task_name, args.model_name, args.seed,
                                                                    args.mtl_task_num))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.to(device)
        # 多少步内验证集的loss没有变小就提前停止
        patience, eval_loss = 0, 0
        # train
        model.train()
        user_avg_weight_in_batch = torch.zeros((self.num_tasks, 1))
        # item_avg_weight_in_batch = torch.zeros((self.num_tasks, 1))
        # save_message = []
        for e in range(epoch):
            y_train_true = collections.defaultdict(list)
            y_train_predict = collections.defaultdict(list)
            total_loss, count_train = 0, 0
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y= x.to(device), y.to(device)
                batch_size = x.shape[0]
                # predict, user_weight, item_weight = model(x)
                predict, u_weight = model(x)
                user_weight = u_weight.detach()
                # user weights处理
                if idx == 0:
                    user_avg_weight_in_batch = torch.mean(user_weight, dim=0).detach()
                else:
                    user_avg_weight_in_batch = user_avg_weight_in_batch*0.9 + 0.1*torch.mean(user_weight, dim=0).detach()
                user_loss_weight = torch.where(y == 0,
                                               torch.sigmoid(torch.div((1-user_avg_weight_in_batch), (1-user_weight) + self.eps)),
                                               torch.sigmoid(torch.div(user_avg_weight_in_batch, user_weight + self.eps)))
                # # user_id转换
                # train_x = x.cpu().numpy()
                # train_x[:, 0] = le['user_id'].inverse_transform(train_x[:, 0].astype(int))
                # train_x[:, 27] = le['video_id'].inverse_transform(train_x[:, 27].astype(int))
                # save_message.append(np.concatenate([train_x, y.cpu().numpy(), predict.cpu().detach().numpy(), user_loss_weight.detach().numpy()], axis=1))
                # item weights处理
                # if idx == 0:
                #     item_avg_weight_in_batch = torch.mean(item_weight, dim=0).detach()
                # else:
                #     item_avg_weight_in_batch = item_avg_weight_in_batch*0.1 + 0.9*torch.mean(item_weight, dim=0).detach()
                # item_loss_weight = torch.where(y == 0,
                #                                torch.sigmoid((1-item_avg_weight_in_batch) / (1 - item_weight)),
                #                                torch.sigmoid(item_avg_weight_in_batch / item_weight))
                # print(user_avg_weight_in_batch, item_avg_weight_in_batch)
                # print(item_loss_weight, user_loss_weight)
                # print(user_weight, item_weight)
                # 计算weight
                # loss_weight = torch.mul(item_loss_weight, user_loss_weight)
                loss_weight_easy = [1, 1, 1, 1, 1, 5]
                # 计算loss
                main_loss = sum(
                    [torch.matmul(user_loss_weight[:, i].T, self.loss_function[i](predict[:, i], y[:, i], reduction='none'))*loss_weight_easy[i] for i in range(self.num_tasks)])
                reg_loss = self.get_regularization_loss()
                aux_loss = sum(
                    [self.loss_function[i](u_weight[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                curr_loss = main_loss + aux_loss + reg_loss
                self.writer.add_scalar("train_loss", float(curr_loss)/batch_size, idx)
                for i, l in enumerate(self.labels):
                    y_train_true[l] += list(y[:, i].cpu().numpy())
                    y_train_predict[l] += list(predict[:, i].cpu().detach().numpy())
                    # self.writer.add_scalar("user_"+l+'_batch_avg', user_avg_weight_in_batch[i], idx)
                optimizer.zero_grad()
                curr_loss.backward()
                optimizer.step()
                total_loss += float(curr_loss)
                count_train += batch_size
            auc = dict()
            for l in self.labels:
                auc[l] = roc_auc_score(y_train_true[l], y_train_predict[l])
                print("Epoch %d train loss is %.3f, %s auc is %.3f" % (e + 1, total_loss / count_train, l, auc[l]))
            # 验证
            total_eval_loss = 0
            model.eval()
            count_eval = 0
            y_val_true = collections.defaultdict(list)
            y_val_predict = collections.defaultdict(list)
            # save_message = []
            for idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                batch_size = x.shape[0]
                # predict, user_weight, item_weight = model(x)
                predict, u_weight = model(x)
                user_weight = u_weight.detach()
                for i, l in enumerate(self.labels):
                    y_val_true[l] += list(y[:,i].cpu().numpy())
                    y_val_predict[l] += list(predict[:, i].cpu().detach().numpy())
                # user weights处理
                user_loss_weight = torch.where(y == 0,
                                               torch.sigmoid(torch.div((1-user_avg_weight_in_batch), (1-user_weight) + self.eps)),
                                               torch.sigmoid(torch.div(user_avg_weight_in_batch, user_weight + self.eps)))
                # item weights处理
                # item_loss_weight = torch.where(y==0,
                #                                torch.sigmoid(((1-item_avg_weight_in_batch) / (1-item_weight)) * -1),
                #                                torch.sigmoid((item_avg_weight_in_batch / item_weight) * -1))
                # print(item_loss_weight, user_loss_weight)
                # 计算weight
                # loss_weight = torch.mul(item_loss_weight, user_loss_weight)
                # # user_id转换
                # val_x = x.cpu().numpy()
                # val_x[:, 0] = le['user_id'].inverse_transform(val_x[:, 0].astype(int))
                # val_x[:, 27] = le['video_id'].inverse_transform(val_x[:, 27].astype(int))
                # save_message.append(np.concatenate([val_x, y.cpu().numpy(), predict.cpu().detach().numpy(), user_loss_weight.detach().numpy()], axis=1))
                # loss计算
                main_loss = sum(
                    [torch.matmul(user_loss_weight[:, i].T, self.loss_function[i](predict[:, i], y[:, i], reduction='none'))*loss_weight_easy[i] for i in range(self.num_tasks)])
                reg_loss = self.get_regularization_loss()
                aux_loss = sum(
                    [self.loss_function[i](u_weight[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                curr_loss = main_loss + aux_loss + reg_loss
                self.writer.add_scalar("val_loss", curr_loss.detach().mean(), idx)
                total_eval_loss += float(curr_loss)
                count_eval += batch_size
            # final_save_message = np.concatenate(save_message, axis=0)
            # val_df = pd.DataFrame(final_save_message)
            # val_df.to_csv(save_data_path+'val_predict_data_mmoeaud.csv', index=False)
            auc = dict()
            for l in self.labels:
                auc[l] = roc_auc_score(y_val_true[l], y_val_predict[l])
                print("Epoch %d val loss is %.3f, %s auc is %.3f" % (e + 1, total_eval_loss / count_eval, l, auc[l]))
            print('-------------------------------')
            # earl stopping
            if e == 0:
                eval_loss = total_eval_loss / count_eval
                state = model.state_dict()
                torch.save(state, path)
            else:
                if total_eval_loss / count_eval < eval_loss:
                    eval_loss = total_eval_loss / count_eval
                    state = model.state_dict()
                    torch.save(state, path)
                else:
                    if patience < early_stop:
                        patience += 1
                    else:
                        print("val loss is not decrease in %d epoch and break training" % patience)
                        break
        # test
        state = torch.load(path)
        model.load_state_dict(state)
        total_test_loss = 0
        model.eval()
        count_test = 0
        y_test_true = collections.defaultdict(list)
        y_test_predict = collections.defaultdict(list)
        save_message = []
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            # predict, user_weight, item_weight = model(x)
            predict, u_weight = model(x)
            user_weight = u_weight.detach()
            for i, l in enumerate(self.labels):
                y_test_true[l] += list(y[:, i].cpu().numpy())
                y_test_predict[l] += list(predict[:, i].cpu().detach().numpy())
            # user weights处理
            user_loss_weight = torch.where(y == 0,
                                           torch.sigmoid(
                                               torch.div((1 - user_avg_weight_in_batch), (1 - user_weight) + self.eps)),
                                           torch.sigmoid(torch.div(user_avg_weight_in_batch, user_weight + self.eps)))
            # item weights处理
            # item_loss_weight = torch.where(y == 0,
            #                                torch.sigmoid(((1 - item_avg_weight_in_batch) / (1 - item_weight)) * -1),
            #                                torch.sigmoid((item_avg_weight_in_batch / item_weight) * -1))
            # 计算weight
            # loss_weight = torch.mul(item_loss_weight, user_loss_weight)
            # user_id转换
            test_x = x.cpu().numpy()
            test_x[:, 0] = le['user_id'].inverse_transform(test_x[:, 0].astype(int))
            test_x[:, 27] = le['video_id'].inverse_transform(test_x[:, 27].astype(int))
            save_message.append(np.concatenate([test_x, y.cpu().numpy(), predict.cpu().detach().numpy(), user_loss_weight.detach().numpy()], axis=1))
            # loss计算
            main_loss = sum(
                [torch.matmul(user_loss_weight[:, i].T, self.loss_function[i](predict[:, i], y[:, i], reduction='none')) *
                 loss_weight_easy[i] for i in range(self.num_tasks)])
            reg_loss = self.get_regularization_loss()
            aux_loss = sum(
                [self.loss_function[i](u_weight[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
            curr_loss = main_loss + aux_loss + reg_loss
            self.writer.add_scalar("test_loss", curr_loss.detach().mean()/batch_size, idx)
            total_test_loss += float(curr_loss)
            count_test += batch_size
        final_save_message = np.concatenate(save_message, axis=0)
        test_df = pd.DataFrame(final_save_message)
        test_df.to_csv(save_data_path + 'test_predict_data_mmoeasd.csv', index=False)
        auc = dict()
        for l in self.labels:
            auc[l] = roc_auc_score(y_test_true[l], y_test_predict[l])
            print("Epoch %d test loss is %.3f, %s auc is %.3f" % (e + 1, total_test_loss / count_test, l, auc[l]))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss