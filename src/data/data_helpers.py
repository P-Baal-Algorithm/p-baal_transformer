from typing import List, Dict
from datasets import ClassLabel, Features, Value, Dataset, DatasetDict


def get_dataset_mapping(
    train: Dataset, valiation_matched: Dataset, test_matched: Dataset
) -> Dict[str, Dataset]:
    """
    Function that converts the huggingface hub dataset split names to the ones use for the p-baal algorithm
    """
    return DatasetDict(
        {"train": train, "valiation_matched": valiation_matched, "test_matched": test_matched}
    )


def set_features(task_name: str, num_classes: int, class_names: List) -> Features:
    """
    Function to set the task specific features for the dataset. Input data for a given task will need to be received in this format

    Returns
    ------------
        Features: Hugginface dataset obect of the features of the data to be loaded
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


def add_index_column(datasets_dict: Dict) -> None:
    """
    Function to add an index column to a hugginface dataset

    Parameters:
    ------------
        datasets_dict: Dictionary of huggingface datasets to be iterated over
    """
    # Iterate over each dataset in the dictionary and add an index column
    for dataset_name, dataset in datasets_dict.items():
        datasets_dict[dataset_name] = dataset.add_column("idx", range(len(dataset)))
