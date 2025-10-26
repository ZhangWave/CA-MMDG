import torch
import torch.nn as nn
import torch.nn.functional as F


class HardTripletLoss(nn.Module):
    """
    硬/最难三元组损失
    (pytorch实现：https://omoindrot.github.io/triplet-loss)

    对于每个锚点，我们获取最难的正样本和最难的负样本来形成一个三元组。
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        参数:
            margin: 三元组损失的边距
            hardest: 如果为True，仅考虑最难的三元组损失。
            squared: 如果为True，输出为成对的平方欧几里得距离矩阵。
                     如果为False，输出为成对的欧几里得距离矩阵。
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        参数:
            labels: 批量的标签，大小为 (batch_size,)
            embeddings: 嵌入张量，形状为 (batch_size, embed_dim)

        返回:
            triplet_loss: 含有三元组损失的标量张量
        """
        # 计算成对的距离
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # 获取最难的正样本对
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # 获取最难的负样本对
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # 将最大d(a, p)和最小d(a, n)结合成最终的三元组损失
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)
        else:
            # 获取锚点-正样本距离和锚点-负样本距离
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # 计算大小为(batch_size, batch_size, batch_size)的3D张量
            # triplet_loss[i, j, k]将包含anc=i, pos=j, neg=k的三元组损失
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()  # 获取三元组掩码
            triplet_loss = loss * mask

            # 移除负损失（即容易的三元组）
            triplet_loss = F.relu(triplet_loss)

            # 计算硬三元组的数量（即triplet_loss > 0）
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            # 计算平均三元组损失
            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    """
    计算所有嵌入之间的2D距离矩阵。

    参数:
        x (Tensor): 形状为 (batch_size, embed_dim) 的嵌入张量
        squared (bool): 如果为True，返回平方欧几里得距离矩阵。如果为False，返回欧几里得距离矩阵。
        eps (float): 在计算平方根时添加的小常数，防止除零错误。

    返回:
        Tensor: 成对的距离矩阵，形状为 (batch_size, batch_size)
    """
    # 计算所有嵌入之间的点积矩阵
    cor_mat = torch.matmul(x, x.t())

    # 获取每个嵌入的平方L2范数。我们可以直接取点积矩阵的对角线。
    # 这也提供了更高的数值稳定性（结果的对角线将恰好为0）。
    norm_mat = cor_mat.diag()

    # 根据公式计算成对距离矩阵:
    # ||a - b||^2 = ||a||^2 - 2 <a, b> + ||b||^2
    # 形状为 (batch_size, batch_size)
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)

    # 由于计算误差，一些距离可能为负数，因此我们将所有距离设置为 >= 0.0
    distances = F.relu(distances)

    if not squared:
        # 由于在 distances == 0.0 时 sqrt 的梯度是无限的（例如：在对角线上）
        # 我们需要在 distances == 0.0 的地方添加一个小的epsilon
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)

        # 修正添加的epsilon：将掩码上的距离设置为正好为0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """
    返回一个2D掩码，其中 mask[a, p] 为 True 当且仅当 a 和 p 是不同的并且具有相同的标签。

    参数:
        labels (Tensor): 标签张量，形状为 (batch_size,)

    返回:
        Tensor: 锚点-正样本掩码，形状为 (batch_size, batch_size)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取一个对角线为False的矩阵（即 indices_not_equal[a, p] 为True 当且仅当 a != p）
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # 检查 labels[i] 是否等于 labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    # 返回一个掩码，其中掩码值为True当且仅当a和p是不同的并且具有相同的标签
    mask = indices_not_equal * labels_equal

    return mask


import torch

def _get_anchor_negative_triplet_mask(labels):
    """
    返回一个2D掩码，其中 mask[a, n] 为 True 当且仅当 a 和 n 具有不同的标签。

    参数:
        labels (Tensor): 标签张量，形状为 (batch_size,)

    返回:
        Tensor: 锚点-负样本掩码，形状为 (batch_size, batch_size)
    """
    # 检查 labels[i] 是否不等于 labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    # 掩码为 True 当且仅当标签不同
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(labels):
    """
    返回一个3D掩码，其中 mask[a, p, n] 为 True 当且仅当三元组 (a, p, n) 有效。

    三元组 (i, j, k) 有效的条件是:
        - i, j, k 彼此不同
        - labels[i] == labels[j] 并且 labels[i] != labels[k]

    参数:
        labels (Tensor): 标签张量，形状为 (batch_size,)

    返回:
        Tensor: 三元组掩码，形状为 (batch_size, batch_size, batch_size)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 检查 i, j 和 k 彼此不同
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # 检查 labels[i] == labels[j] 并且 labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    # 合并两个掩码
    mask = distinct_indices * valid_labels

    return mask

