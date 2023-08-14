# SequenceDataLoader for Convolutional LSTM Regression Tasks

When training video prediction models using Keras, handling large images that need to be split into smaller regions for effective training becomes crucial. The SequenceDataLoader presented here is designed to simplify this process by seamlessly extracting smaller regions from larger images on-the-fly. This enables the training of neural networks for regression tasks, facilitating accurate video prediction.

## Overview

The primary goal of the SequenceDataLoader is to streamline the process of preparing training data for video prediction models. It achieves this by dynamically extracting smaller regions from larger images, creating an efficient pipeline for neural network training. The loader is particularly useful when working with datasets where images need to be partitioned into manageable regions for analysis.

## Input Data Structure

The loader expects an input DataFrame structured as follows:

| tile_id  | region_id | target | region_coordinates   |
|----------|-----------|--------|----------------------|
| 4879217  | 1225      | 0.123    | (581, 806, 626, 846) |
| 4879218  | 1225      | 0.423    | (626, 806, 671, 846) |
| 4879219  | 1225      | 0.143    | (671, 806, 716, 846) |
| 4879220  | 1225      | 0.032    | (716, 806, 761, 846) |
| 4879221  | 1225      | 0.213    | (761, 806, 805, 846) |

Here, `tile_id` corresponds to the unique identifier for each tile, `region_id` represents the larger image region containing the tiles, `target` holds the target value for regression, and `region_coordinates` specifies the coordinates of the region within the larger image.

## Image Storage Structure

The images corresponding to `region_id` are stored locally in the following format:

```
Images/
|-- label_id/
|   |-- region_id.tif
```

You can easily customize the file extension directly within the class to suit your data.

## Data Extraction Pipeline

The SequenceDataLoader's data extraction pipeline is illustrated in the following diagram:

![DataExtractionPipeline](https://github.com/victor-radermecker/SequenceDataLoader/blob/main/img/diagram.png?raw=true)

This pipeline showcases how the loader operates by sequentially extracting smaller regions from the labeled images based on the provided coordinates. These regions are then utilized for neural network training to predict video sequences accurately.

In essence, the SequenceDataLoader serves as a valuable tool for simplifying the data preparation process when training video prediction models. By seamlessly managing the extraction of smaller regions from larger images, it empowers the seamless training of neural networks for regression tasks, ultimately enhancing the accuracy of video predictions.


## Example Usage

### Dynamic World Model Overview

The dynamic world model leverages the power of deep learning, particularly ConvLSTM networks, to capture temporal patterns in satellite images and make predictions about urbanization rates. The model takes a sequence of satellite images from multiple years (2016 to 2022) for a specific geographical region and learns to identify patterns and changes in urbanization over time.

### Key Parameters

- `IMG_SIZE`: The dimensions of the input images (64x64 pixels).
- `BATCH_SIZE`: The number of image sequences to include in each training batch (256).
- `N_CHANNELS`: The number of color channels in the images (3 for RGB).
- `LABELS`: List of years (2016 to 2022) for which image data is available.
- `IMG_DIR`: Directory path containing the training image data.
- `RANDOM_SEED`: Seed value for random number generation to ensure reproducibility (42).
- `TARGET_TYPE`: Type of target for prediction ("target" for urbanization rate of 2022).

```python
IMG_SIZE = (64, 64)
BATCH_SIZE = 256
N_CHANNELS = 3
LABELS = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
IMG_DIR = "../Images/Train"
RANDOM_SEED = 42
TARGET_TYPE = "target" # "last-image"

# Initialize the SequenceDataLoader for training data
data_loader = SequenceDataLoader(
    labels=LABELS,
    data=DATA,
    image_dir=IMG_DIR,
    dim=IMG_SIZE,
    batch_size=BATCH_SIZE,
    n_channels=N_CHANNELS,
    random_seed=RANDOM_SEED,
    shuffle=True,
    normalize=True
)
```

### Data Loading and Processing

The `SequenceDataLoader` is initialized to load and process the training data. It loads the sequence of satellite images corresponding to the specified years, applies necessary preprocessing steps such as resizing and normalization, and creates batches of data for training. The data loader utilizes the provided `LABELS`, `IMG_DIR`, `IMG_SIZE`, `BATCH_SIZE`, `N_CHANNELS`, and `RANDOM_SEED` parameters to efficiently manage the data loading process.

The provided image ![DataExtractionPipeline](https://github.com/victor-radermecker/SequenceDataLoader/blob/main/img/example.png?raw=true) illustrates the data extraction pipeline. It depicts the process of loading and preparing the sequence of satellite images for training the dynamic world model.



![DataExtractionPipeline](https://github.com/victor-radermecker/SequenceDataLoader/blob/main/img/example.png?raw=true)

Explain that we have loaded images of dynamic world for one specific region from 2016 to 2022 and that we are training a convolutional LSTM on these sequences to predict the urbanization rate of 2022. 


## Note

The provided example is a basic guide on how to use the `SequenceDataLoader` class. Depending on your specific use case and requirements, you may need to customize certain methods or add additional functionalities to suit your project's needs.

Always ensure that the class parameters, data paths, and preprocessing steps are set up correctly to achieve the desired behavior and performance.
