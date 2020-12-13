def dice(input, target):
    b = input.size(0)
    binary_input = (input > 0.5).float()

    intersect = (binary_input * target).view(b, -1).sum(dim=-1)
    union = binary_input.view(b, -1).sum(dim=-1) + target.view(b, -1).sum(dim=-1)
    score = 2 * intersect / (union + 1e-3)

    return score.mean().item()


def recall(input, target):
    b = input.size(0)
    binary_input = (input > 0.5).float()

    true_positives = (binary_input.view(b, -1) * target.view(b, -1)).sum(dim=-1)
    all_positives = target.view(b, -1).sum(dim=-1)
    recall = true_positives / all_positives

    return recall.mean().item()


def precision(input, target):
    b = input.size(0)
    binary_input = (input > 0.5).float()

    true_positives = (binary_input.view(b, -1) * target.view(b, -1)).sum(dim=-1)
    all_positive_calls = binary_input.view(b, -1).sum(dim=-1)
    precision = true_positives / all_positive_calls

    return precision.mean().item()
