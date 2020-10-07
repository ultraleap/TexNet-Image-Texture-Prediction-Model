"""Ultraleap Image Texture Prediction Model © Ultraleap Limited 2020

Licensed under the Ultraleap closed source licence agreement; you may not use this file except in compliance with the License.

A copy of this License is included with this download as a separate document. 

Alternatively, you may obtain a copy of the license from: https://www.ultraleap.com/closed-source-licence/

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License."""

from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Nadam, SGD


class TexNetModels(object):
    """The TexNet model class can be instantiated in order to initialise the different networks (CNN & MLP) for training
    on our different data types. In addition we can run the training step from this object too.
    """

    def __init__(self, **kwargs):
        """Initialise the TexNet model.

        Parameters
        ----------
        kwargs: dictionary of params.
            If the user wishes to prepare and train a model with the various different input features then this
            dict object can be used to state which parameters they would like to use.
        """
        self.train_i = kwargs.get('image_training')
        self.train_m = kwargs.get('matrix_training')
        self.train_h = kwargs.get('haralick_training')

        self.train_i_data = kwargs.get('train_image_data')
        self.train_m_data = kwargs.get('train_matrix_data')
        self.train_h_data = kwargs.get('train_haralick_data')

        self.test_i_data = kwargs.get('test_image_data')
        self.test_m_data = kwargs.get('test_matrix_data')
        self.test_h_data = kwargs.get('test_haralick_data')

        self.val_i_data = kwargs.get('val_image_data')
        self.val_m_data = kwargs.get('val_matrix_data')
        self.val_h_data = kwargs.get('val_haralick_data')

        self.train_t = kwargs.get('train_target')
        self.test_t = kwargs.get('test_target')
        self.val_t = kwargs.get('val_target')

    def texnet_conv2d(self, input_data):
        """ Architecture of the CNN model that can be trained on 2D input data (images and matrices).
        Architecture is 3 Keras Conv2D layers with window size of 7x7 and 3x3, 16 filters, 'relu' activations,
        and He normal kernel initialisation. Max pooling layers between each Conv2D layer, window size 4x4 and 2x2.
        Finally, a Dense layer with L2 kernel regularisation set at 0.005.

        Parameters
        ----------
        input_data: array_like
            Expects some 2D matrix input of shape 256x256.

        Returns
        -------
        texnet_conv2D: Keras Convolutional model.
        """
        _input = Input(shape=input_data[0].shape)  # Set input shape from length of first dimension in matrix input.

        # Initialise layers
        texnet_conv2D = Conv2D(16, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(_input)
        texnet_conv2D = MaxPooling2D(pool_size=(4, 4))(texnet_conv2D)
        texnet_conv2D = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            texnet_conv2D)
        texnet_conv2D = MaxPooling2D(pool_size=(2, 2))(texnet_conv2D)
        texnet_conv2D = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            texnet_conv2D)
        texnet_conv2D = MaxPooling2D(pool_size=(2, 2))(texnet_conv2D)

        # Flatten and initialise Dense layer with 16 filters and L2 kernal regularisation.
        texnet_conv2D = Flatten()(texnet_conv2D)
        texnet_conv2D = Dense(16, activation='relu', kernel_regularizer=l2(0.005))(texnet_conv2D)
        texnet_conv2D = Model(inputs=_input, outputs=texnet_conv2D)
        return texnet_conv2D

    def texnet_mlp(self, input_data):
        """ Architecture of the Multi-Layer Perceptron that can be trained on Haralick feature data.
        Architecture is 2 Keras Dense layers with 16 filters, 'relu' activations,
        and He normal kernel initialisation. The second Dense layer features L2 kernel regularisation set at 0.01.

        Parameters
        ----------
        input_data: array_like
            Expects some 1D NumPy array.

        Returns
        -------
        tex_net_conv2D: Keras Convolutional model.
        """
        _input = Input(shape=(input_data.shape[1],))
        texnet_mlp = Dense(16, activation='relu')(_input)
        texnet_mlp = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(texnet_mlp)
        texnet_mlp = Model(inputs=_input, outputs=texnet_mlp)
        return texnet_mlp

    def prepare_model(self):
        """ This function initialises the model architecture based on the selected input features.
        If only Haralick data is used then only the TexNetMLP model will be initialised. If more than one input feature
        is used then the model will be ensembled with a Keras concatenate layer before predictions are done via a final
        Dense layer.

        Returns
        -------
        model: Compiled model either single network (CNN or MLP), or some ensembled model comprised of both model
        architectures.
        """
        model_inputs = []
        model_outputs = []

        # Initialise TexNet Conv2D model if images are given as input.
        if self.train_i:
            image_model = self.texnet_conv2d(self.train_i_data)
            model_inputs.append(image_model.input)
            model_outputs.append(image_model.output)

        # Initialise TexNet Conv2D model if matrices are given as input.
        if self.train_m:
            matrix_model = self.texnet_conv2d(self.train_m_data)
            model_inputs.append(matrix_model.input)
            model_outputs.append(matrix_model.output)

        # Initialise TexNet MLP if Haralick features are given as input.
        if self.train_h:
            haralick_model = self.texnet_mlp(self.test_h_data)
            model_inputs.append(haralick_model.input)
            model_outputs.append(haralick_model.output)

        # If only a singular input feature is used then ensembling is not conducted.
        if len(model_outputs) == 1:
            final_layer = Dense(1, activation='sigmoid')(model_outputs[0])
            model = Model(inputs=model_inputs, outputs=final_layer)
        else:
            concatenate_model = concatenate(model_outputs)
            final_layer = Dense(len(model_outputs), activation='relu')(concatenate_model)
            final_layer = Dense(1, activation='sigmoid')(final_layer)
            model = Model(inputs=model_inputs, outputs=final_layer)

        if len(model_outputs) == 0:
            print(
                "Error: No models have been trained!"
                "Please set a training data set to 'True' so a model can be correctly compiled.")
        else:
            return model

    def train_model(self, tensor, epochs, batch_size=None, loss_function='mae', optimizer='adam'):
        """ This object will conduct the training step of the compiled model obtained from the 'prepare_model' function.

        Parameters
        ----------
        tensor: Tensorflow Keras model.
            Input should be the prepared model obtained from the 'prepare_model' function.
        epochs: int
            Number of epochs model should run through. Default = 150.
        batch_size: int
            How many different features should be passed through the model at one time. Default = 1.
        loss_function: string
            Which loss function the model should be minimising for. Default is Mean Absolute Error (mae).
        optimizer: string
            Which optimiser the model should be using during training.
            Default = 'adam' - Keras Nadam (Adam with Nesterov Momentum) lr = 0.0005.

        Returns
        -------
        train_data: array_like
            The split set of training data used during model training
        test_data: array_like
            The split set of test data
        val_data: array_list
            The split set of data for validation.
        tensor.fit: Tf Keras model
            Trained model over the input number of epochs.
        """
        if self.train_i is None and self.train_m is None and self.train_h is None:
            print("\n")
            print("No data as input! Select a training data set.")

        # Initialise optimizer
        if optimizer == 'adam':
            optimizer = Nadam(lr=0.0005)
        if optimizer == 'sgd':
            optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.6)

        # Compile the model with the selected optimiser and loss function.
        tensor.compile(optimizer=optimizer, loss=loss_function)

        train_data = []
        test_data = []
        val_data = []

        if self.train_i:
            train_data.append(self.train_i_data)
            test_data.append(self.test_i_data)
            val_data.append(self.val_i_data)

        if self.train_m:
            train_data.append(self.train_m_data)
            test_data.append(self.test_m_data)
            val_data.append(self.val_m_data)

        if self.train_h:
            train_data.append(self.train_h_data)
            test_data.append(self.test_h_data)
            val_data.append(self.val_h_data)

        if batch_size is None:
            batch_size = len(self.train_i_data)

        # Return output trained model.
        return train_data, test_data, val_data, tensor.fit(train_data, self.train_t, batch_size=batch_size,
                                                           epochs=epochs, verbose=1,
                                                           validation_data=(val_data, self.val_t))