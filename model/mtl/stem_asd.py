
import collections
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from model.layers.inputs import varlen_embedding_lookup, get_varlen_pooling_list, stem_varlen_embedding_lookup, \
    stem_get_varlen_pooling_list
from ..layers.attention import SelfAttentionModule
from ..layers.core import DNN, PredictionLayer
from sklearn.metrics import roc_auc_score
save_data_path = './dataset/data/save/'

class STEM_Layer(nn.Module):
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim, expert_hidden_units,
                 gate_hidden_units, hidden_activations, l2_reg_dnn, dnn_dropout, dnn_use_bn, init_std, device):
        super(STEM_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.shared_experts = nn.ModuleList([DNN(input_dim, expert_hidden_units, activation=hidden_activations,
                                                 l2_reg=l2_reg_dnn,
                                                 dropout_rate=dnn_dropout,
                                                 use_bn=dnn_use_bn,
                                                 init_std=init_std,
                                                 device=device) for _ in range(self.num_shared_experts)])
        self.specific_experts = nn.ModuleList([nn.ModuleList([DNN(input_dim,expert_hidden_units,
                                                                  activation=hidden_activations,
                                                                  l2_reg=l2_reg_dnn,
                                                                  dropout_rate=dnn_dropout,
                                                                  use_bn=dnn_use_bn,
                                                                  init_std=init_std,
                                                                  device=device)
                                                              for _ in range(self.num_specific_experts)])
                                               for _ in range(num_tasks)])
        self.gate = nn.ModuleList([DNN(input_dim, gate_hidden_units,
                                       activation=hidden_activations,
                                       l2_reg=l2_reg_dnn,
                                       dropout_rate=dnn_dropout,
                                       use_bn=dnn_use_bn,
                                       init_std=init_std,
                                       device=device) for _ in range(self.num_tasks + 1)])
        self.gate_final_layer = nn.ModuleList([nn.Linear(gate_hidden_units[-1],
                                                         num_specific_experts * num_tasks + num_shared_experts,
                                                         bias=False)
                                               for _ in range(self.num_tasks+1)])
        self.gate_activation = nn.Softmax(dim=-1)

    def forward(self, x, return_gate=False):
        """
        x: list, len(x)==num_tasks+1
        """
        specific_expert_outputs = []
        shared_expert_outputs = []
        # specific experts
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)
        # shared experts
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))

        # gate
        stem_outputs = []
        stem_gates = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                # for specific experts
                gate_input = []
                for j in range(self.num_tasks):
                    if j == i:
                        gate_input.extend(specific_expert_outputs[j])
                    else:
                        specific_expert_outputs_j = specific_expert_outputs[j]
                        specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                        gate_input.extend(specific_expert_outputs_j)
                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input, dim=1)  # (?, num_specific_experts*num_tasks+num_shared_experts, dim)
                gate_mid = self.gate[i](x[i] + x[-1])
                gate_output = self.gate_final_layer[i](gate_mid)
                gate = self.gate_activation(gate_output)  # (?, num_specific_experts*num_tasks+num_shared_experts)
                if return_gate:
                    specific_gate = gate[:, :self.num_specific_experts * self.num_tasks].mean(0)
                    task_gate = torch.chunk(specific_gate, chunks=self.num_tasks)
                    specific_gate_list = []
                    for tg in task_gate:
                        specific_gate_list.append(torch.sum(tg))
                    shared_gate = gate[:, -self.num_shared_experts:].mean(0).sum()
                    target_task_gate = torch.stack(specific_gate_list + [shared_gate], dim=0).view(-1)  # (num_task+1,1)
                    assert len(target_task_gate) == self.num_tasks + 1
                    stem_gates.append(target_task_gate)
                stem_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)  # (?, dim)
                stem_outputs.append(stem_output)
            else:
                # for shared experts
                gate_input = []
                for j in range(self.num_tasks):
                    gate_input.extend(specific_expert_outputs[j])
                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input, dim=1)  # (?, num_specific_experts*num_tasks+num_shared_experts, dim)
                gate_mid = self.gate[i](x[i] + x[-1])
                gate_output = self.gate_final_layer[i](gate_mid)
                gate = self.gate_activation(gate_output)  # (?, num_specific_experts*num_tasks+num_shared_experts)
                stem_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)  # (?, dim)
                stem_outputs.append(stem_output)

        if return_gate:
            return stem_outputs, stem_gates
        else:
            return stem_outputs

class STEMASD(nn.Module):
    def __init__(self, categorical_feature_dict, continuous_feature_dict, var_cat_feature_dict, user_features, labels, writer, emb_dim=128,
                 expert_dnn_hidden_units=(256, 128), num_shared_experts=1, num_specific_experts=1, num_layers=2,
                 gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,),
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
        super(STEMASD, self).__init__()
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
        self.num_layers = num_layers
        self.categorical_feature_dict = categorical_feature_dict
        self.continuous_feature_dict = continuous_feature_dict
        self.var_cat_feature_dict = var_cat_feature_dict
        self.user_features = user_features
        self.num_tasks = len(labels)
        self.writer = writer
        self.labels = labels
        self.task_types = ['binary']*self.num_tasks
        self.embedding_dim = emb_dim
        self.eps = torch.FloatTensor([1e-8]).to(device)
        # TODO
        self.loss_function = [F.binary_cross_entropy]*self.num_tasks
        # print(len(self.categorical_feature_dict), len(self.continuous_feature_dict), len(self.history_feature_dict))

        if device:
            self.device = device

        # embedding初始化
        self.embedding_dict = nn.ModuleDict(
            {feat: nn.Embedding(num[0], emb_dim*(self.num_tasks+1), sparse=False) # 非one-hot编码
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
        # 1. expert + gate
        self.stem_layers = nn.ModuleList([STEM_Layer(num_shared_experts, num_specific_experts, self.num_tasks,
                                                     input_dim= self.input_dim if i==0 else expert_dnn_hidden_units[-1],
                                                     expert_hidden_units= expert_dnn_hidden_units,
                                                     gate_hidden_units=gate_dnn_hidden_units,
                                                     hidden_activations=dnn_activation,
                                                     l2_reg_dnn=l2_reg_dnn,
                                                     dnn_dropout=dnn_dropout,
                                                     dnn_use_bn=dnn_use_bn,
                                                     init_std=init_std,
                                                     device=device) for i in range(self.num_layers)])
        # 2. 任务塔
        self.tower = nn.ModuleList([DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation,
                                        l2_reg=l2_reg_dnn,
                                        dropout_rate=dnn_dropout,
                                        use_bn=dnn_use_bn,
                                        init_std=init_std,
                                        device=device)
                                    for _ in range(self.num_tasks)])
        # 直接默认塔网络有多层
        self.self_attention = SelfAttentionModule(tower_dnn_hidden_units[-1], self.num_tasks)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1, bias=False)
                                                    for _ in range(self.num_tasks)])
        # 3. user tower (task-specific)
        self.user_tower_dnn = nn.ModuleList(
            [DNN(self.user_input_dim, tower_dnn_hidden_units, activation=dnn_activation,
                 l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                 init_std=init_std, device=device) for _ in range(self.num_tasks)])
        self.user_tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1,
            bias=False) for _ in range(self.num_tasks)])

        # 4. 输出经过激活函数
        self.output_activation = nn.ModuleList([self.get_output_activation(self.task_types[i]) for i in range(self.num_tasks)])
        self.user_out = nn.ModuleList([self.get_output_activation(self.task_types[i]) for i in range(self.num_tasks)])

        regularization_modules = [self.stem_layers, self.tower, self.tower_dnn_final_layer, self.user_tower_dnn, self.user_tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        # self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)

    def forward(self, x):
        # assert x.size()[1] == len(self.categorical_feature_dict) + len(self.continuous_feature_dict) + len(self.var_cat_feature_dict)*20+1
        # embedding
        cat_embed_list, con_embed_list, user_cat_embed_list = [[] for _ in range(self.num_tasks+1)], list(), [[] for _ in range(self.num_tasks+1)]
        for cat_feature, num in self.categorical_feature_dict.items():
            cat_emb = self.embedding_dict[cat_feature](x[:, num[1]].long())
            feature_embs = cat_emb.split(self.embedding_dim, dim=1)
            for i in range(self.num_tasks+1):
                cat_embed_list[i].append(feature_embs[i])
                if cat_feature in self.user_features:
                    user_cat_embed_list[i].append(feature_embs[i])


        for con_feature, num in self.continuous_feature_dict.items():
            con_embed_list.append(x[:, num[1]].unsqueeze(1))

        sequence_embed_dict = stem_varlen_embedding_lookup(x, self.embedding_dict, self.var_cat_feature_dict, self.embedding_dim, self.num_tasks)
        varlen_embed_list = stem_get_varlen_pooling_list(sequence_embed_dict, x, self.var_cat_feature_dict, self.num_tasks, self.device)

        # embedding 融合
        cat_embed = [torch.cat(cat_embed_list[i], axis=1) for i in range(self.num_tasks+1)]
        con_embed = torch.cat(con_embed_list, axis=1)
        varelen_embed = [torch.cat(varlen_embed_list[i], axis=1) for i in range(self.num_tasks+1)]

        # 用户辅助网络 embedding concat
        user_cat_embed = [torch.cat(user_cat_embed_list[i], axis=1) for i in range(self.num_tasks+1)]

        # hidden layer
        dnn_input = [torch.cat([cat_embed[i], con_embed, varelen_embed[i]], axis=1).float() for i in range(self.num_tasks+1)]  # batch * hidden_size
        user_dnn_input = [torch.cat([user_cat_embed[i], varelen_embed[i]], axis=1).float() for i in range(self.num_tasks+1)]

        for i in range(self.num_layers):
            stem_outputs = self.stem_layers[i](dnn_input)
            dnn_input = stem_outputs
        user_weights = []
        for i, label in enumerate(self.labels):
            # 计算当前任务辅助网络输入
            emp_label = x[:, self.continuous_feature_dict['emp_'+label][1]].unsqueeze(1)# .detach()
            curr_user_dnn_input = torch.cat([user_dnn_input[i], emp_label], axis=1).float()
            user_tower_dnn_out = self.user_tower_dnn[i](curr_user_dnn_input)
            user_tower_dnn_logit = self.user_tower_dnn_final_layer[i](user_tower_dnn_out)
            # user辅助塔结果处理
            user_output = self.user_out[i](user_tower_dnn_logit)
            user_weights.append(user_output)

        task_dnn_outs = [self.tower[i](stem_outputs[i]) for i in range(self.num_tasks)]
        attention_input = torch.stack(task_dnn_outs, dim=1)
        attention_output = self.self_attention(attention_input)
        task_final_layer_input = torch.split(attention_output + attention_input, 1, dim=1)
        task_outs = []
        for i, label in enumerate(self.labels):
            tower_dnn_logit = self.tower_dnn_final_layer[i](task_final_layer_input[i].squeeze(1))
            # 主塔结果处理
            output = self.output_activation[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        user_weights = torch.cat(user_weights, -1)
        return task_outs, user_weights

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
        for e in range(epoch):
            y_train_true = collections.defaultdict(list)
            y_train_predict = collections.defaultdict(list)
            total_loss, count = 0, 0
            for idx, (x, y) in tqdm(enumerate(train_loader)):
                x, y= x.to(device), y.to(device)
                predict, user_weight = model(x)
                # user weights处理
                if idx == 0:
                    user_avg_weight_in_batch = torch.mean(user_weight, dim=0).detach()
                else:
                    user_avg_weight_in_batch = user_avg_weight_in_batch * 0.9 + 0.1 * torch.mean(user_weight,
                                                                                                 dim=0).detach()
                user_loss_weight = torch.where(y == 0,
                                               torch.tanh(torch.div((1 - user_avg_weight_in_batch),
                                                                    (1 - user_weight) + self.eps)),
                                               torch.tanh(torch.div(user_avg_weight_in_batch, user_weight + self.eps)))
                loss_weight = user_loss_weight
                loss_weight_easy = [1, 1, 1, 10, 10, 10]
                # 计算loss
                loss = sum(
                    [torch.matmul(loss_weight[:, i].T, self.loss_function[i](predict[:, i], y[:, i], reduction='none'))*loss_weight_easy[i] for i in range(self.num_tasks)])
                reg_loss = self.get_regularization_loss()
                curr_loss = loss + reg_loss
                self.writer.add_scalar("train_loss", float(curr_loss), idx)
                for i, l in enumerate(self.labels):
                    y_train_true[l] += list(y[:, i].cpu().numpy())
                    y_train_predict[l] += list(predict[:, i].cpu().detach().numpy())
                optimizer.zero_grad()
                curr_loss.backward()
                optimizer.step()
                total_loss += float(curr_loss)
                count += 1
            auc = dict()
            for l in self.labels:
                auc[l] = roc_auc_score(y_train_true[l], y_train_predict[l])
                print("Epoch %d train loss is %.3f, %s auc is %.3f" % (e + 1, total_loss / count, l, auc[l]))
            # 验证
            total_eval_loss = 0
            model.eval()
            count_eval = 0
            y_val_true = collections.defaultdict(list)
            y_val_predict = collections.defaultdict(list)
            save_message = []
            for idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                predict, user_weight = model(x)
                for i, l in enumerate(self.labels):
                    y_val_true[l] += list(y[:, i].cpu().numpy())
                    y_val_predict[l] += list(predict[:, i].cpu().detach().numpy())
                # user weights处理
                user_loss_weight = torch.where(y == 0,
                                               torch.tanh(torch.div((1 - user_avg_weight_in_batch),
                                                                    (1 - user_weight) + self.eps)),
                                               torch.tanh(torch.div(user_avg_weight_in_batch, user_weight + self.eps)))
                loss_weight = user_loss_weight
                loss_weight_easy = [1, 1, 1, 1, 10, 10]
                loss = sum(
                    [torch.matmul(loss_weight[:, i].T, self.loss_function[i](predict[:, i], y[:, i], reduction='none'))*loss_weight_easy[i] for i in range(self.num_tasks)])
                reg_loss = self.get_regularization_loss()
                curr_loss = loss + reg_loss
                self.writer.add_scalar("val_loss", curr_loss.detach().mean(), idx)
                total_eval_loss += float(curr_loss)
                count_eval += 1
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
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            predict, user_weight = model(x)
            for i, l in enumerate(self.labels):
                y_test_true[l] += list(y[:, i].cpu().numpy())
                y_test_predict[l] += list(predict[:, i].cpu().detach().numpy())
            # user weights处理
            user_loss_weight = torch.where(y == 0,
                                           torch.tanh(
                                               torch.div((1 - user_avg_weight_in_batch), (1 - user_weight) + self.eps)),
                                           torch.tanh(torch.div(user_avg_weight_in_batch, user_weight + self.eps)))
            loss_weight = user_loss_weight
            # user_id转换
            test_x = x.cpu().numpy()
            test_x[:, 0] = le['user_id'].inverse_transform(test_x[:, 0].astype(int))
            test_x[:, 27] = le['video_id'].inverse_transform(test_x[:, 27].astype(int))
            save_message.append(np.concatenate(
                [test_x, y.cpu().numpy(), predict.cpu().detach().numpy(), user_loss_weight.detach().numpy()], axis=1))
            # loss计算
            loss = sum(
                [torch.matmul(loss_weight[:, i].T, self.loss_function[i](predict[:, i], y[:, i], reduction='none')) for
                 i in range(self.num_tasks)])
            reg_loss = self.get_regularization_loss()
            curr_loss = loss + reg_loss
            self.writer.add_scalar("test_loss", curr_loss.detach().mean(), idx)
            total_test_loss += float(curr_loss)
            count_test += 1
        final_save_message = np.concatenate(save_message, axis=0)
        test_df = pd.DataFrame(final_save_message)
        test_df.to_csv(save_data_path+'test_predict_data_stemasd.csv', index=False)
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

    def get_output_activation(self, task):
        if task == "binary":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))