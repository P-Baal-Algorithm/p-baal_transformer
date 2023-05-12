from typing import List, Dict
from datasets import ClassLabel, Features, Value, Dataset


def get_dataset_mapping(
    train: Dataset, valiation_matched: Dataset, test_matched: Dataset
) -> Dict[str, Dataset]:
    """
    Function that converts the huggingface hub dataset split names to the ones use for the p-baal algorithm
    """
    return {"train": train, "valiation_matched": valiation_matched, "test_matched": test_matched}


def set_features(task_name: str, num_classes: int, class_names: List) -> Features:
    """
    Function to set the task specific features for the dataset. Input data for a given task will need to be received in this format

    Returns
    ------------
        Features: Hugginface dataset obect of the features of the data to be loadedpyt
    """
    if task_name == "MNLI":
        features_dict = {
            "hypothesis": Value(dtype="string", id=None),
            "idx": Value(dtype="int64", id=None),
            "label": ClassLabel(num_classes=num_classes, names=class_names, id=None),
            "premise": Value(dtype="string", id=None),
        }
    elif task_name == "CLASS":  # this name should potentially be changed
        features_dict = {
            "corpus": Value(dtype="string", id=None),
            "label": ClassLabel(num_classes=num_classes, names=class_names, id=None),
        }
    features = Features(features_dict)

    return features
