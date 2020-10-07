"""Ultraleap Image Texture Prediction Model © Ultraleap Limited 2020

Licensed under the Ultraleap closed source licence agreement; you may not use this file except in compliance with the License.

A copy of this License is included with this download as a separate document. 

Alternatively, you may obtain a copy of the license from: https://www.ultraleap.com/closed-source-licence/

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License."""

import numpy as np
import pandas as pd
import sklearn.metrics as skm
import matplotlib.pyplot as plt


class VisualisePerformance(object):
    """ This class offers a few of the visualisation functions that display the models performance.
    """

    def __init__(self, **kwargs):
        """ Initialise the object and provide the input data for display.

        kwargs: dict
            Provide a dictionary of the inputs selected for visualisation.
        """
        self.train_d = kwargs.get('train_data')
        self.test_d = kwargs.get('test_data')
        self.val_d = kwargs.get('val_data')
        self.train_t = kwargs.get('train_target')
        self.test_t = kwargs.get('test_target')
        self.val_t = kwargs.get('val_target')

    def predictions(self, compiled_model, batch_size=None):
        """ This function will compute the predicted values based on the set of
        test data passed to the class during initialisation.

        Parameters
        ----------
        compiled model: Tf Keras compiled model.
            This should be the model that was trained on output from the 'train_model' function in the 'TexNetModels'
            class.
        """
        return compiled_model.predict(self.test_d, batch_size=batch_size, verbose=1)

    def display_loss(self, prepared_model):
        """ Displays the various loss values obtained from the predictions made on train, test, and validation data
        sets.

        Parameters
        ----------
        prepared model: Tf Keras model.
            This should be the model that was trained on output from the 'prepare_model' function in the 'TexNetModels'
            class.
        """
        train_loss = prepared_model.evaluate(self.train_d, self.train_t, verbose=0)
        test_loss = prepared_model.evaluate(self.test_d, self.test_t, verbose=0)
        val_loss = prepared_model.evaluate(self.val_d, self.val_t, verbose=0)
        return ('Train Loss: {}'.format(train_loss),
                'Validation Loss: {}'.format(val_loss),
                'Test Loss: {}'.format(test_loss))

    def compute_accuracy_metrics(self, predicted):
        """ Displays the various errors quantities by comparing actual values with predictions made by the model.

        Parameters
        ----------
        predicted: array-like
            The predicted values returned from the 'predictions' function.

        Returns
        -------
        MAPE: float64
            Mean Absolute Percentage Error
        MAE: float64
            Mean Absolute Error
        MSE: float64
            Mean Squared Error
        RMSE: float64
            Root Mean Squared Error
        R2: float64
            R2 - Coefficient of determination.
        """
        mape = np.mean(np.abs(self.test_t.values[:, np.newaxis] - predicted / self.test_t.values[:, np.newaxis]))
        mae = skm.mean_absolute_error(self.test_t.values, predicted)
        mse = skm.mean_squared_error(self.test_t.values, predicted)
        rmse = np.sqrt(mse)
        r2 = skm.r2_score(self.test_t.values, predicted)
        return {'MAPE': mape, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    def plot_loss(self, tensor):
        """ Plots loss over epochs for both training and validation data.

        Parameters
        ----------
        tensor: Tf Keras trained model.
            This should be the output trained model from 'train_model' function.

        Returns
        -------

        Plot of loss over epoch.
        """
        plt.plot(tensor.history['loss'])
        plt.plot(tensor.history['val_loss'])
        plt.title('Model Loss over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    def create_prediction_df(self, predicted):
        """ Generate a Pandas DataFrame that contains the associated texture dimension actual values and predicted values.

        Parameters
        ----------
        predicted: array_like
            The predicted values returned from the 'predictions' function.

        Returns
        -------
        prediction_df: Pandas DataFrame
        """
        prediction_df = pd.DataFrame(self.test_t)
        prediction_df['predicated_target'] = predicted
        prediction_df = prediction_df.sort_values(by=prediction_df.columns[0], ascending=True)

        return prediction_df

    def plot_predictions(self, data_frame, dimension):
        """ Generate a plot of predicted values against actual values.

        Parameters
        ----------

        data_frame: Pandas DataFrame
            Values obtained from 'create_prediction_df' function
        dimension: string
            Type of texture dimension plot is being computed for.
        """
        plt.style.use(u'seaborn-whitegrid')

        xdata = data_frame.index.format()
        ydata = data_frame.iloc[:, 0]
        y2data = data_frame.iloc[:, 1]

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.plot(xdata, ydata, 'o--', alpha=0.7, label="Subjective {} Values".format(dimension), c='b')
        for x, y in zip(xdata, ydata):
            label = "{:.2f}".format(y)
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 2), ha='center')

        ax.plot(xdata, y2data, 'o-', alpha=0.7, label="Model Predicted {} Values".format(dimension), c='g')
        for x, y in zip(xdata, y2data):
            label = "{:.2f}".format(y)
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -2), ha='center')

        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Subjective {}: Actual Values vs Model Predicted Values'.format(dimension))  # Set title.
        ax.set_xlabel('Image Texture', fontsize=18)
        ax.set_ylabel('{}'.format(dimension), fontsize=18)
        ax.margins(x=0.01)
        plt.ylim(0, 1)

        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.show()