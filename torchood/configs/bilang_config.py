from pathlib import Path

"""
The get_config function returns a dictionary that contains various configuration parameters for a model.
These parameters include the batch size, number of epochs, learning rate, sequence length, model dimensions, source and target languages, model folder and basename, preloading flag, tokenizer file name, and experiment name.
"""


def get_config():
    return {
        "batch_size": 2048,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "proload": True,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


"""
This function get_weights_file_path takes two inputs:
config (a dictionary) and epoch (a string).
It returns the file path of the weights file based on the given configuration and epoch.
"""

"""
## Example USES
config = {
"model_folder": "models",
"model_basename": "model_"
}
epoch = "10"
file_path = get_weights_file_path(config, epoch)
print(file_path)
"""


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
