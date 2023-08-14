# SequenceDataLoader Readme

The `SequenceDataLoader` class was designed for generating data in batches suitable for training and prediction using the Keras deep learning framework. This class extends the `Sequence` class from TensorFlow's Keras API, making it efficient for working with large datasets and ensuring smooth training processes. The class is designed to facilitate the preparation and loading of data for image classification tasks involving multi-label inputs, where each input is associated with multiple labels.

## Goal

Create a dataloader to train video predictions models using Keras. Sometimes, you have large images that you need to split into smaller ones for training. 
The goal of this dataloader is to extract smaller regions of a larger images on-the-fly.

Ouput:
    - Sequences of images X
    - Target Variable Y





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
