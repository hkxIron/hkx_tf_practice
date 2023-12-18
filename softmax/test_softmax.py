import torch
from torch import scatter

# logits:[batch, class]
# target:[batch]
def my_cross_entropy(logits, target, reduction="mean"):
    # 减去最大值，以免溢出
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits = logits - logits_max

    exp = torch.exp(logits)  # 这里对input所有元素求exp
    # gather:直接获取指定index的值, 而不需要计算所有分子
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()  # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
    tmp2 = exp.sum(1)  # 在exp第一维求和，这是softmax的分母
    softmax = tmp1 / tmp2  # softmax公式：ei / sum(ej)

    # cross-entropy公式： -yi * log(pi) 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，公式中的pi就是softmax的结果
    log = -torch.log(softmax)
    # 这里的reduce是针对样本batch
    # 官方实现中，reduction有mean/sum及none，只是对交叉熵后处理的差别
    if reduction == "mean":
        return log.mean()
    else:
        return log.sum()


import torch.nn.functional as F

N,C=3,5
logits = torch.randn(N,C, requires_grad=True)
target = torch.randint(high=C, size=(N,), dtype=torch.int64)
print(f"logits:{logits} target:{target}")

loss1_mean = F.cross_entropy(logits, target)
loss2_mean = my_cross_entropy(logits, target)

print(loss1_mean)  # tensor(1.5247, grad_fn=<NllLossBackward>)
print(loss2_mean)  # tensor(1.5247, grad_fn=<MeanBackward0>)


def test_scatter():
    n, c = 5,3
    one_hot = torch.zeros(
        (n, c),
        device='cpu',
        requires_grad=False
    )
    one_hot.scatter_(1, target.unsqueeze(dim=-1), 1)

    # scatter one-hot vectors into GPUs
    one_hot = scatter(one_hot, dim=-1)
    print(one_hot)

test_scatter()