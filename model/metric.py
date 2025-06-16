# standard libraries

# third party libraries
import torch
from torch import Tensor

# local libraries


__all__ = [
    "accuracy",
    "top_k_acc",
]


def accuracy(output: Tensor, target: Tensor) -> float:
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output: Tensor, target: Tensor, k: int = 3) -> float:
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
