"""Ultraleap Image Texture Prediction Model © Ultraleap Limited 2020

Licensed under the Ultraleap closed source licence agreement; you may not use this file except in compliance with the License.

A copy of this License is included with this download as a separate document. 

Alternatively, you may obtain a copy of the license from: https://www.ultraleap.com/closed-source-licence/

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License."""

import os
import argparse
import numpy as np
import pandas as pd
from feature_lists import FeaturesLists
from prepare_data import PrepareData
from texnet_models import TexNetModels
from visualise_performance import VisualisePerformance


def main(file, image_dir, image_size, test_texture_list, predictor_variable, epochs, batch_size,
         distances, angles, image_training, haralick_training, matrix_training):

    # Check if file or data frame passed to function.
    if isinstance(file, pd.DataFrame):
        data_frame = file
    if not os.path.exists(file):
        print("The file %s does not exist!" % file)
    else:
        data_frame = pd.read_csv(r"{}".format(file), error_bad_lines=False)  # Create data frame from file.

    # Initialise our FeaturesLists object.
    feature_creation = FeaturesLists(image_dir, image_size=image_size)

    # Generate our list of images.
    image_list = feature_creation.create_image_list()

    # Create list of matrices, obtain the optimal matrix input values, generate our Haralick features for each image.
    matrices, inputs, haralick = feature_creation.create_matrix_list(image_list,
                                                                     distances,
                                                                     angles)

    # Initialise our Prepare Data object and pass our list of test textures that we want to
    # retain for evaluating the performance of our model.
    prepare_data = PrepareData(test_texture_list)

    # We initialise our data in a new data frame including additional columns containing our computed feature sets.
    df = prepare_data.append_feature_data(data_frame=data_frame,
                                          image_list=image_list,
                                          matrix_list=matrices,
                                          haralick_list=haralick)

    # We calculate the Log of cluster shade and prominence to better display the range of computed values.
    for col in [col for col in df.columns if 'har_cls' in col]:
        df["log_{}".format(col)] = np.log(np.abs(df[col]))

    # Values in Haralick data are then rescaled between 0 and 1 if their max is greater than 1.
    for col in [col for col in df.columns if 'har_' in col and df[col].max() > 1]:
        df[col] = prepare_data.scale_data(df[col], 0, 1)

    # Define which features we wish to use during training.
    HARALICK_FEATURES = ['har_homo', 'har_corr', 'har_contrast', 'har_energy',
                         'har_mean', 'har_stdev', 'log_har_cls_prom', 'log_har_cls_shade']

    # Create each of the separate train, test, and validation data sets.
    train, test, val = prepare_data.split_for_training(df)

    # Reshape the image and matrix data so that it can be passed to our model.
    train_images = prepare_data.reshape_array(train.image_list, (len(train.image_list),
                                                                 train.image_list[0].shape[0],
                                                                 train.image_list[0].shape[1], 1))
    test_images = prepare_data.reshape_array(test.image_list, (len(test.image_list),
                                                               test.image_list[0].shape[0],
                                                               test.image_list[0].shape[1], 1))

    val_images = prepare_data.reshape_array(val.image_list, (len(val.image_list),
                                                             val.image_list[0].shape[0],
                                                             val.image_list[0].shape[1], 1))

    train_matrix = prepare_data.reshape_array(train.matrix_list, (len(train.matrix_list),
                                                                  train.matrix_list[0].shape[0],
                                                                  train.matrix_list[0].shape[1], 1))
    test_matrix = prepare_data.reshape_array(test.matrix_list, (len(test.matrix_list),
                                                                test.matrix_list[0].shape[0],
                                                                test.matrix_list[0].shape[1], 1))

    val_matrix = prepare_data.reshape_array(val.matrix_list, (len(val.matrix_list),
                                                              val.matrix_list[0].shape[0],
                                                              val.matrix_list[0].shape[1], 1))

    # Initialise our dictionary that defines which parameters we want to train on.
    features = {
        'image_training': image_training,
        'matrix_training': matrix_training,
        'haralick_training': haralick_training,
        'train_image_data': train_images,
        'train_matrix_data': train_matrix,
        'train_haralick_data': train.loc[:, HARALICK_FEATURES],
        'test_image_data': test_images,
        'test_matrix_data': test_matrix,
        'test_haralick_data': test.loc[:, HARALICK_FEATURES],
        'val_image_data': val_images,
        'val_matrix_data': val_matrix,
        'val_haralick_data': val.loc[:, HARALICK_FEATURES],
        'train_target': train[predictor_variable] / 100,  # Divide predictor variable values by 100
        'test_target': test[predictor_variable] / 100,
        'val_target': val[predictor_variable] / 100
    }

    # Initilise our TexNetModels object and pass to it our feature key word argument dictionary.
    texNet = TexNetModels(**features)

    # Prepare the model for training
    prepared_model = texNet.prepare_model()

    # Train our model, and produce the split train, test, and validation data sets.
    train_data, test_data, val_data, tensor_history = texNet.train_model(prepared_model,
                                                                         epochs,
                                                                         batch_size)

    # Initiliase key word args for visualising model performance.
    visualisation_data = {
        'train_data': train_data,
        'test_data': test_data,
        'val_data': val_data,
        'train_target': train[predictor_variable] / 100,  # Divide predictor variable values by 100
        'test_target': test[predictor_variable] / 100,
        'val_target': val[predictor_variable] / 100
    }

    # Initialise the VisualisePerformance object and pass to it the dictionary of key word params.
    performance_visualiser = VisualisePerformance(**visualisation_data)

    # Generate predictions from our trained model.
    predictions = performance_visualiser.predictions(prepared_model)

    # Display loss values for each of our data sets.
    print(performance_visualiser.display_loss(prepared_model))

    # Display accuracy metrics for predictions by our model.
    print(performance_visualiser.compute_accuracy_metrics(predictions))

    # Display loss function over no. of epochs.
    performance_visualiser.plot_loss(tensor_history)

    # Create a Pandas Data Frame that contains our predictions and real values in ascending order.
    predictions_df = performance_visualiser.create_prediction_df(predictions)

    # Plot the performance of our predictions as a line graph in comparison to our actual values.
    performance_visualiser.plot_predictions(predictions_df, predictor_variable)

    print('Model trained using: Image Data - {}, Matrix Data - {}, Haralick Data - {}'.format(image_training,
                                                                                              matrix_training,
                                                                                              haralick_training))


# Arguments to be passed from command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a Tensorflow model whose inputs are image, GLCM, and Haralick feature data.')

    parser.add_argument('--file',
                        dest='file',
                        metavar='Data Frame .csv file.',
                        required=True,
                        help='The path to the filtered .csv file for the appropriate predictor variable')

    parser.add_argument('--image_dir',
                        dest='image_dir',
                        metavar='Image file path location.',
                        required=True,
                        help='The path to the image list.')

    parser.add_argument('--image_size',
                        dest='image_size',
                        metavar='Image size for processing.',
                        required=True, type=int,
                        help='Integer image size for processing.')

    parser.add_argument('--test_texture_list',
                        dest='test_texture_list',
                        nargs='*', type=str,
                        default=None,
                        metavar='List of strings of image texture names.',
                        help='Provide a list of strings of image names should you require a specific test data set.')

    parser.add_argument('--predictor_variable',
                        dest='predictor_variable',
                        metavar='Subjective texture dimension to predict.',
                        required=True,
                        help='Input image texture predictor variable. Choose from: roughness, bumpiness, stickiness, '
                             'warmness, hardness. Append mean or median to name for either values: "dimension_mean".')

    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        default='150',
                        metavar='No. of epochs for model to train over.',
                        help='An integer value for number of epochs.')

    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default='1',
                        metavar='Model batch size to use.',
                        help='An integer value for batch size. Default = 1.')

    parser.add_argument('--distances',
                        dest='distances',
                        nargs='*', type=int,
                        default=[1, 2, 3, 4, 5, 8, 10, 12, 15, 20],
                        metavar='List of distance values to use for calculating pixel co-occurrences matrices.',
                        help='Integer list of distance values for GLCM.')

    parser.add_argument('--angles',
                        dest='angles',
                        nargs='*', type=int,
                        default=[0, 45, 90, 135],
                        metavar='List of angles in degrees to calculate matrices.',
                        help='Integer list of angle values in degrees for GLCM.')

    parser.add_argument('--image_training_on',
                        dest='image_training',
                        action='store_true')
    parser.add_argument('--image_training_off',
                        dest='image_training',
                        action='store_false')
    parser.set_defaults(image_training=True)

    parser.add_argument('--haralick_training_on',
                        dest='haralick_training',
                        action='store_true')
    parser.add_argument('--haralick_training_off',
                        dest='haralick_training',
                        action='store_false')
    parser.set_defaults(haralick_training=True)

    parser.add_argument('--matrix_training_on',
                        dest='matrix_training',
                        action='store_true')
    parser.add_argument('--matrix_training_off',
                        dest='matrix_training',
                        action='store_false')
    parser.set_defaults(matrix_training=True)

    args = parser.parse_args()

    main(args.file,
         args.image_dir,
         args.image_size,
         args.test_texture_list,
         args.predictor_variable,
         args.epochs,
         args.batch_size,
         args.distances,
         args.angles,
         args.image_training,
         args.haralick_training,
         args.matrix_training)