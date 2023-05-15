from typing import Dict, List, Union
from src.data.data_helpers import set_features, get_dataset_mapping, add_index_column
from src.settings import DATA_FILES
from datasets import load_dataset


def data_loader(
    type_file: str,
    file_directory: str,
    task_name: str,
    num_classes: int = None,
    class_names: List[Union[int, str]] = None,
    **kwargs
):
    """
    Function to load the data from directory or online huggingface dataset

    Parameters
    ------------
        type_file (str):
            the type of file to be loaded, Options:
                    - 'huggingface': Loads an online dataset from huggingface
                    - 'csv': Reads csv data from file_directory
                    - 'json' : Reads json data from file_directory
        file_directory (str):
            if typefile is huggingface then name of online dataset, else directory in which the datasets are contained
        task_name (str):
            the type of NLP task that is being performed
                - 'MNLI' - natural language inference task with premise/hypothesis setting
                - 'CLASS' - standard text classification task
        num_classes (str), Optional:
            number of output classes in dataset (Default = None)
        class_names (List[Union[int, str]]), Optional:
            list of output labels given in either strings or integers. len(class_names) must always be equal to num_classes
                Example: class_names = ['entailment', 'neutral']

    Returns
    ------------
        Dataset : huggingface NLP dataset
    """

    if type_file == "huggingface":
        dataset = load_huggingface_dataset(file_directory)
    else:
        dataset = load_custom_dataset(
            type_file, file_directory, task_name, num_classes, class_names
        )
    return dataset


def load_huggingface_dataset(file_directory: str):
    """
    Function to load dataset that has been uploaded to huggingface

    file_directory (str):
        Huggingface dataset name
            - "rotten_tomatoes" - rotten tomatoes dataset
    """
    dataset = load_dataset(file_directory)
    if file_directory == "rotten_tomatoes":
        out_dataset = get_dataset_mapping(dataset["train"], dataset["validation"], dataset["test"])

    add_index_column(out_dataset)
    return out_dataset


def load_custom_dataset(
    type_file: str, file_directory: str, task_name: str, num_classes: str, class_names: str
) -> Dict:
    """
    Function to load a custom dataset contained in a local directory.
    """
    features = set_features(task_name, num_classes, class_names)

    dataset = load_dataset(
        type_file,
        data_dir=file_directory,
        data_files=DATA_FILES,
        features=features,
        encoding="cp1252",
    )
    return dataset
