
# BLoad: Enhancing Neural Network Training with Efficient Sequential Data Handling

Welcome to the official repo for [BLoad: Enhancing Neural Network Training with Efficient Sequential Data Handling](https://arxiv.org/abs/2310.10879). We provide an example of using our strategy on top of the [Action Genome: Actions as Composition of Spatio-temporal Scene Graphs](https://arxiv.org/abs/1912.06992).

## Overview

The repo has three main components:

- **AG.py**: (Action Genome) Dataset Class: A foundational class for loading and preprocessing image data.
- **AG_BLoad.py**: An extension of the AG dataset class, adopting the BLoad strategy.
- **test_loader.py**: A utility script for demonstrating dataset loading and iterating through the data in batches.

## Installation

Before using the toolkit, ensure you have the following dependencies installed:
- PyTorch
- Pillow
- NumPy
- tqdm

You can install these packages using pip.

```
pip install torch pillow numpy tqdm
```

## AG (Action Genome) Dataset Class

The AG class is the core dataset class for loading and preprocessing data. It supports reading image frames and corresponding annotations, applying transformations, and preparing the dataset for model training or evaluation.

## AG_BLoad

AG_BLoad implements the method described in the white paper. It uses a dictionary containing the number of frames as keys and the video name as a list of values.

### Features

- **Block Generation**: Organizes videos into blocks based on frame counts for efficient loading.
- **Dynamic Randomization**: Ensures diversity in training and evaluation by changing block composition each epoch.
- **Batch Processing**: Custom collate function to group data into batches, handling padding and resets as necessary.

## Usage

This method can be applied to several datasets which contain different length input, be it videos, audios, etc.

## Running the Test Loader

Execute the script with necessary arguments.

```
python test_loader.py --AG_path "path/to/Action Genome/dataset" --max_size_defined 800
```

For the Action Genome dataset, it is expected to be in the following structure:
```
root/
├── frames/
│   ├── video1/
│   ├── video2/
│   ├── video3/
└── annotations/
    ├── AG_HOIA_train_sgdet.pkl
    ...
```


This will load the dataset, create a DataLoader, and iterate through the dataset in batches, printing progress with tqdm.

## Contributing

Contributions to improve the toolkit are welcome. Please follow standard GitHub practices for submitting pull requests.
