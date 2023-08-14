# SequenceDataLoader for Video Prediction Model Training

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

## Key Features

1. **Efficient Batch Generation**: The `SequenceDataLoader` efficiently generates batches of data for training or prediction, reducing memory overhead by loading data on-the-fly.

2. **Customizable Data Loading**: This class is designed to handle datasets consisting of images, associated labels, and tabular data. It offers the flexibility to load images and tabular data, enabling more complex and diverse data preparation.

3. **Region-Based Batching**: The class supports region-based batching, where batches are formed based on specified regions and their corresponding tiles. This is particularly useful for scenarios where data needs to be grouped based on spatial relationships.

4. **Data Augmentation**: By modifying the `_load_region` method, data augmentation techniques can be integrated into the data loading process to improve model generalization.

5. **Multi-Label Support**: The class supports multi-label classification tasks, allowing each input image to be associated with multiple labels simultaneously.

## How to Use

1. **Initialization**: Create an instance of the `SequenceDataLoader` class by providing the necessary parameters such as `labels`, `list_IDs`, `target`, `tile_region_dic`, `tile_coordinates`, `image_dir`, `dim`, `batch_size`, `n_channels`, and other optional parameters.

2. **Data Generation**: The `__getitem__` method generates one batch of data containing both input data (`X`) and target labels (`y`). If tabular data is available, it can also be included in the output.

3. **Region-Based Batching**: The `create_batches` method groups data into batches based on specified regions and tile associations. This region-based batching strategy helps maintain spatial relationships in the data.

4. **Data Loading**: The `_load_region` method handles the loading and preprocessing of image data for each region and associated tiles. This method can be customized to include data augmentation techniques or other preprocessing steps.

5. **Data Normalization**: The class provides an option to normalize image data if required, by setting the `normalize` parameter to `True`.

6. **Data Augmentation (Optional)**: To apply data augmentation techniques, modify the `_load_region` method to perform transformations such as rotations, flips, or color adjustments.

## Example Usage

```python
# Initialize SequenceDataLoader
data_loader = SequenceDataLoader(
    labels=['label1', 'label2'],
    list_IDs=list_of_ids,
    target=target_data,
    tile_region_dic=tile_region_mapping,
    tile_coordinates=tile_coordinates,
    image_dir='path_to_images',
    dim=(128, 128),
    batch_size=32,
    n_channels=3,
    shuffle=True,
    tab_data=tabular_data,
    normalize=True
)

# Iterate through batches and train your model
for epoch in range(epochs):
    for batch_X, batch_y in data_loader:
        # Train your model using batch_X and batch_y

# For prediction, use the same data loader
predictions = model.predict(data_loader)

# Access tabular data for each batch
for batch_X, batch_y in data_loader:
    batch_tab_data = batch_X[1]  # Assuming tab_data was provided during initialization
```

## Note

The provided example is a basic guide on how to use the `SequenceDataLoader` class. Depending on your specific use case and requirements, you may need to customize certain methods or add additional functionalities to suit your project's needs.

Always ensure that the class parameters, data paths, and preprocessing steps are set up correctly to achieve the desired behavior and performance.
