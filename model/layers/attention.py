import torch
import torch.nn as nn

class SelfAttentionModule(nn.Module):
    def __init__(self, input_dim, task_num):
        super(SelfAttentionModule, self).__init__()
        self.task_num = task_num
        self.input_dim = input_dim

        # 定义用于self-attention的权重矩阵
        self.W_q = nn.Parameter(torch.randn(input_dim, input_dim))
        self.W_k = nn.Parameter(torch.randn(input_dim, input_dim))
        self.W_v = nn.Parameter(torch.randn(input_dim, input_dim))

        # 使用 nn.init 初始化参数
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, input_vectors):
        # 输入向量的维度应为 (batch_size, task_num, input_dim)
        batch_size, _, _ = input_vectors.size()

        # 使用self-attention计算权重
        Q = torch.matmul(input_vectors, self.W_q)
        K = torch.matmul(input_vectors, self.W_k)
        V = torch.matmul(input_vectors, self.W_v)

        # 计算attention分数
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))

        # 计算softmax得到权重
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        # 使用权重对V进行加权求和得到self-attention结果
        attention_output = torch.matmul(attention_weights, V)

        return attention_output

# 示例用法
task_num = 5
input_dim = 128

# 创建SelfAttentionModule实例
self_attention_module = SelfAttentionModule(input_dim, task_num)

# 生成示例输入
batch_size = 3
input_vectors = torch.randn(batch_size, task_num, input_dim)

# 调用forward方法进行self-attention计算
output = self_attention_module(input_vectors)

print("Input shape:", input_vectors.shape)
print("Output shape:", output.shape)
