from torch import nn, tensor


def get_weighted_loss(weights, num_classes = 10):
    if weights is None:
        weights = tensor([0.0] * num_classes)

    return weights, nn.CrossEntropyLoss(weights)

# class CustomNLLLoss(nn.NLLLoss):
#     def __init__(self, weight=None, size_average=None, ignore_index=-100,
#                  reduce=None, reduction='mean'):
#         super(CustomNLLLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
#         if weight is not None:
#             self.weight.requires_grad = True