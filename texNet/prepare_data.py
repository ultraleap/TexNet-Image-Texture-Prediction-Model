"""Ultraleap Image Texture Prediction Model © Ultraleap Limited 2020

Licensed under the Ultraleap closed source licence agreement; you may not use this file except in compliance with the License.

A copy of this License is included with this download as a separate document. 

Alternatively, you may obtain a copy of the license from: https://www.ultraleap.com/closed-source-licence/

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PrepareData(object):
    """This object class can be instantiated in order to conduct various splitting and rescaling requirements before our
    model is to be trained.
    """

    def __init__(self, test_texture_list=None, train_size=0.9, shuffle=True):
        """Initialise the class for the purpose of preparing our data.

        Parameters
        ----------
        test_texture_list: array_like(str), optional
            If the user would like to split the data set to exclude a specific set of images as a test set for model
            prediction, a list of corresponding string values can be provided. This must be the specific name for each
            texture. The initial data frame created from the DataProcessing.ipynb should be queried for these names.
        train_size: float, optional
            If a specific training data size is to be used then a value by which the input data should be split can be
            set.
        shuffle: bool, optional
            Sets whether data should be shuffled during the splitting into train, test and validation sets.
        """
        self.test_list = test_texture_list
        self.train_size = train_size
        self.shuffle = shuffle

    def append_feature_data(self, **args):
        """This function will concatenate the initial data frame from DataProcessing.ipynb with any of the additional
        feature data created in the previous steps, such as image arrays, matrices, haralick features.

        Parameters
        ----------
        data_frame: Pandas Dataframe, optional
            This should be the data frame obtained from running DataProcessing.ipynb.
        image_list: array_like, optional
            A list of corresponding images that should be processed by the model.
        matrix_list: array_like, optional
            This list should contain each matrix that has been computed for any image.
        haralick_list: array_like, optional
            A list of float64 values that contains the computed Haralick features for any of the images in the data set.

        Returns
        -------
        appended_df: Pandas Dataframe
            A concatenated data frame that contains the appended feature lists passed as input.
        """
        data_frame = args.get('data_frame')
        image_list = args.get('image_list')
        matrix_list = args.get('matrix_list')
        haralick_list = args.get('haralick_list')

        if data_frame is not None:
            features_df = pd.DataFrame()
        if image_list is not None:
            features_df['image_list'] = [image / 255 for image in image_list]
        if matrix_list is not None:
            features_df['matrix_list'] = matrix_list
        if haralick_list is not None:
            # Create Pandas DataFrame with columns specific for each Haralick feature.
            features_df['har_homo'] = [item[0] for item in haralick_list]
            features_df['har_contrast'] = [item[2] for item in haralick_list]
            features_df['har_energy'] = [item[3] for item in haralick_list]
            features_df['har_corr'] = [item[4] for item in haralick_list]
            features_df['har_mean'] = [item[5] for item in haralick_list]
            features_df['har_stdev'] = [item[6] for item in haralick_list]
            features_df['har_cls_shade'] = [item[7] for item in haralick_list]
            features_df['har_cls_prom'] = [item[8] for item in haralick_list]

        appended_df = pd.concat([data_frame, features_df], axis=1)
        appended_df.set_index('tex_name', drop=True, inplace=True)
        return appended_df

    def scale_data(self, data, rescale_min, rescale_max):
        """Helper function that will rescale any input data to a specified range.

        Parameters
        ----------
        data: array_like, Pandas Series
            A column from a Pandas DataFrame whose values should be rescaled.
        rescale_min, rescale_max: int
            Integer values by which the range of values in the 'data' input should be rescaled to.

        Returns
        -------
        rescaled_data: array_like, Pandas Series
            The rescaled column of data ranging from some given min - max value range.
        """
        col_max = data.max()
        col_min = data.min()
        return data.apply(lambda x: ((x - col_min) / (col_max - col_min)) * (rescale_max - rescale_min) + rescale_min)

    def scale_df(self, df, rescale_min, rescale_max):
        """Helper function to rescale all columns in a Pandas dataframe if this is required.

        Parameters
        ----------
        df: array_like, Pandas DataFrame
            A complete Pandas DataFrame whose values should be rescaled.
        rescale_min, rescale_max: int
            Integer values by which the range of values in the 'df' input should be rescaled to.

        Returns
        -------
        new_df: Pandas DataFrame
            A new data frame containing rescaled column data ranging from some given min - max value range.
        """
        new_df = pd.DataFrame(df, copy=True)
        for i in range(len(new_df.columns)):
            if new_df.iloc[:, i].dtypes == 'float64':
                new_df.iloc[:, i] = new_df.iloc[:, i].apply(
                    lambda x: ((x - new_df.iloc[:, i].min()) / (new_df.iloc[:, i].max() - new_df.iloc[:, i].min())) *
                              (rescale_max - rescale_min) + rescale_min)
        return new_df

    def reshape_array(self, data, shape):
        """Helper function that will reshape a given set of input data to some alternative shape.

        Parameters
        ----------
        data: array_like
            Some list or Pandas Series of data that should reshaped.
        shape: tuple(int)
            A particular shape by which the input data should be converted into.

        Returns
        -------
        reshaped_data: array_like
            The converted data based on the shape value given.
        """
        data = np.array(data.tolist())
        return np.reshape(data, newshape=shape)

    def split_for_training(self, data):
        """This function will split data input into separate train, validation and test sets.

        Parameters
        ----------
        data: array_like
            Pandas DataFrame of input data.

        Returns
        -------
        train, test, val: split Pandas data frames.
        """
        test, val = None, None

        if self.test_list is not None:
            train = data.drop(index=self.test_list)
            test = data.drop(train.index)
            train, val = train_test_split(train, test_size=len(self.test_list), shuffle=self.shuffle)
        else:
            train, test = train_test_split(data, test_size=1 - self.train_size, shuffle=self.shuffle)
            train, val = train_test_split(train, test_size=len(test), shuffle=self.shuffle)

        return train, test, val