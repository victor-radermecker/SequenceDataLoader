import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import Sequence
import os


class SequenceDataLoader(Sequence):
    def __init__(
        self,
        labels: list,
        data: pd.DataFrame,
        image_dir: str,
        dim: tuple,
        batch_size: int = 32,
        n_channels: int = 1,
        random_seed: int = 42,
        shuffle: bool = True,
        normalize: bool = False,
    ):
        """
        Initializes a SequenceDataLoader instance.

        :param labels: List of each sequence labels. Example: 2016, 2017, 2018, ...
        :param data: Pandas DataFrame containing relevant data.
        :param image_dir: Directory path to the location of images.
        :param dim: Tuple indicating the image dimensions (height, width).
        :param batch_size: Batch size at each iteration.
        :param n_channels: Number of image channels.
        :param random_seed: Seed for random number generation.
        :param shuffle: Whether to shuffle label indexes after each epoch.
        :param normalize: Whether to normalize image pixel values to [0, 1].
        """
        self.labels = labels
        self.data = data
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.normalize = normalize
        self._preprocess()
        self._init_params()
        self.on_epoch_end()
        np.random.seed(random_seed)

    def _init_params(self):
        """
        Initializes the parameters related to data dimensions.
        """
        self.N_regions = len(self.tile_region_dic)
        self.N_tiles = len(self.tile_coordinates)
        self.N_labels = len(self.labels)

    def _preprocess(self):
        """
        Preprocesses the data and extracts relevant information.
        """
        self.list_IDs = self.data["region_id"].unique()
        self.tile_region_dic = (
            self.data[self.data["label_id"] == self.labels[-1]]
            .groupby("region_id")["tile_id"]
            .apply(list)
            .to_dict()
        )
        self.target = self.data[self.data["label_id"] == self.labels[-1]][
            ["tile_id", "target"]
        ].set_index("tile_id")
        self.tile_coordinates = self.data[self.data["label_id"] == self.labels[-1]][
            ["tile_id", "region_coordinates"]
        ].set_index("tile_id")

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        :return: The number of batches per epoch.
        """
        self.nbr_batches = int(np.ceil(self.N_tiles / self.batch_size))
        return self.nbr_batches

    def __getitem__(self, index):
        """
        Generates one batch of data.

        :param index: Index of the batch.
        :return: X and y when fitting. X only when predicting.
        """
        batch = self.batches[index]
        X = self._generate_X(batch)
        y = self._generate_y(batch)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.region_indexes = np.arange(len(self.list_IDs))
        self.indexes_to_regionIDs = dict(zip(self.region_indexes, self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.region_indexes)
        self.create_batches()

    def create_batches(self):
        """
        Creates batches of tiles for each region based on a specified batch size.

        This method generates a list of batches, where each batch is a dictionary with
        region IDs as keys and lists of corresponding tiles as values. Each tile ID
        appears only once across all the batches. The method iterates through the
        specified region IDs in the order provided in 'region_indexes' and assigns
        tiles to each region until a batch reaches the maximum batch size.
        """
        batches = []
        current_batch = {}
        current_batch_size = 0

        for region_id in self.list_IDs:
            tiles = self.tile_region_dic[region_id]

            for tile_id in tiles:
                if tile_id not in current_batch.values():
                    current_batch.setdefault(region_id, []).append(tile_id)
                    current_batch_size += 1

                    if current_batch_size >= self.batch_size:
                        batches.append(current_batch)
                        current_batch = {}
                        current_batch_size = 0

        if current_batch:
            batches.append(current_batch)

        print("Success. Completed creating batches.")
        self.batches = batches

    def _generate_y(self, batch):
        """
        Generates the target variable for a given batch.

        :param batch: Dictionary of region IDs and corresponding tile IDs.
        :return: Numpy array of target values.
        """
        regionLengths = [len(v) for v in batch.values()]
        y = np.empty(sum(regionLengths))
        cursor = 0

        for regionID, tileIDs in batch.items():
            for tileID in tileIDs:
                y[cursor] = self.target.loc[tileID]
                cursor += 1

        return y

    def _generate_X(self, batch):
        """
        Generates a batch of images for the given batch.

        :param batch: Dictionary of region IDs and corresponding tile IDs.
        :return: Numpy array of images.
        """
        regionLengths = [len(v) for v in batch.values()]
        X = np.empty((sum(regionLengths), len(self.labels), *self.dim, self.n_channels))
        cursor = 0

        for regionID, tileIDs in batch.items():
            X[cursor : cursor + len(tileIDs), :] = self._load_region(regionID, tileIDs)
            cursor += len(tileIDs)

        return X

    def _load_region(self, regionID, tileIDs):
        """
        Loads a region from the image directory and processes it.

        :param regionID: Region ID.
        :param tileIDs: List of tile IDs in the region.
        :return: Numpy array of processed images.
        """
        N = len(tileIDs)
        X = np.empty((N, self.N_labels, self.dim[0], self.dim[1], self.n_channels))

        for i, label in enumerate(self.labels):
            region_path = os.path.join(
                self.image_dir,
                str(label),
                "Final",
                f"landcover_batchID_{regionID}.tif",
            )
            img = Image.open(region_path)

            if self.n_channels == 1:
                img = img.convert("L")

            img = np.array(img, dtype="uint8")

            for j, tileID in enumerate(tileIDs):
                coordinates = eval(self.tile_coordinates.loc[tileID].values[0])
                sub_img = self._crop_image(img, tileID, regionID, coordinates)
                if self.normalize:
                    sub_img = np.array(sub_img) / 255.0
                X[j, i, :, :, :] = np.array(sub_img).reshape(
                    self.dim[0], self.dim[1], self.n_channels
                )

        return X

    def _crop_image(self, image, tile_id, batch_id, coordinates):
        """
        Crops and resizes an image to match the specified dimensions.

        :param image: Image array.
        :param tile_id: Tile ID.
        :param batch_id: Batch (region) ID.
        :param coordinates: Coordinates for cropping.
        :return: Processed image.
        """
        xmin, ymin, xmax, ymax = coordinates
        subimage = image[ymin:ymax, xmin:xmax]

        if subimage.shape[0] < 30 or subimage.shape[1] < 30:
            raise Exception(
                f"Tile {tile_id} in Region {batch_id} is too small. Please check the regions and tiles."
            )

        subimage = Image.fromarray(subimage)
        subimage = subimage.resize((self.dim[0], self.dim[1]))
        return subimage
