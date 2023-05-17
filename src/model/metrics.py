import torch


def f1_score_metric(predictions, references):
    # Convert predictions and references to PyTorch tensors
    predictions = torch.tensor(predictions)
    references = torch.tensor(references)

    # Calculate the F1 score using PyTorch functions
    true_positives = torch.sum(predictions * references)
    precision = true_positives / torch.sum(predictions)
    recall = true_positives / torch.sum(references)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score.item()  # Convert the tensor value to a Python float
