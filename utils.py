from scipy.signal import savgol_filter
from sklearn.base import TransformerMixin
# import plotly
# import plotly.graph_objs as go
import json
import pickle
import tensorflow as tf
import pandas as pd

def save_to_disk(ls, fname):
    with open(fname, "wb") as fp:
        pickle.dump(ls, fp)

def read_from_disk(fname):
    with open(fname, "rb") as fp:
        ls = pickle.load(fp)
    return ls

def pickle_file(preprocess_path, file, name):
    #create a pickle file
    picklefile = open(preprocess_path / f'input.nir.{name}', 'wb')
    #pickle the dictionary and write it to file
    pickle.dump(file, picklefile)
    #close the file
    picklefile.close()
    
    return

def json_save(sg_num_points1, sg_num_points2, model_save_path):

    save_dict = {

        "Inputs": [
            {
                "Name": "nir",
                "Regex": "^amp.*$",
                "Pre-processing": [
                    {
                        "Name": "Downsample",
                        "Params": {
                            "step": 2
                        }
                    },
                    {
                        "Name": "SavitzkyGolay",
                        "Params": {
                            "polynomial_order": 2,
                            "derivative_order": 1,
                            "num_points": sg_num_points1,
                        }
                    }
                ]
            },
            {
                "Name": "nir2",
                "Regex": "^amp.*$",
                "Pre-processing": [
                    {
                        "Name": "Downsample",
                        "Params": {
                            "step": 3
                        }
                    },
                    {
                        "Name": "SavitzkyGolay",
                        "Params": {
                            "polynomial_order": 2,
                            "derivative_order": 1,
                            "num_points": sg_num_points2
                        }
                    }
                ]
            }
        ],
    }
    
    with open(model_save_path.parent.parent / 'model.json', 'w') as outfile:
        json.dump(save_dict, outfile)
        
    return

class TrainingPlotly(tf.keras.callbacks.Callback):

    def __init__(self, plot_save_path):
        self.plot_save_path = plot_save_path

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and MSEs
        self.losses = []
        self.mean_squared_error= []
        self.val_losses = []
        self.val_mean_squared_error= []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and MSEs to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=self.losses,
                                mode='lines',
                                name='train_loss'))
            fig.add_trace(go.Scatter(y=self.val_losses,
                                mode='lines',
                                name='val_loss'))
            fig = fig.update_layout(title=f'Training Loss [Epoch {epoch}]', title_x=0.5, xaxis={'title':f'Epoch {epoch}'}, yaxis={'title':'Loss'})
            fig.write_image(str(self.plot_save_path / f'{epoch}.png'))

class DownsamplePy(TransformerMixin):
    """
    Filter which gets every Nth column from a matrix, starting at a given index.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        allowed_keys = ["start_index", "step"]
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        
        if not hasattr(self, "start_index"):
            self.start_index = 0
        if not hasattr(self, "step"):
            self.step = 1
    
    def fit(self, x):
        # self.original_data = x.copy(deep=True)
        # self.original_heads = [float(x) for x in list(x)]
        # self.original_values = x.values
        # self.original_index = x.index.values
        
        self.columns = [i for i in range(self.start_index, x.shape[1], self.step)]
        self.result = x.iloc[:, self.columns].copy(deep=True)
        return self

    def transform(self, df):
    
        return self.result

# ds = DownsamplePy(step=2)

# ds.fit(train_spc_df)

# ds.columns

# ds.transform(train_spc_df)

# ds.fit_transform(train_spc_df).shape




class SavGol(TransformerMixin):
    """
    Description
    -----------
    Savitsky-Golay Filter
    Publication of note
    --------------------
    Fourier transforms in the computation of self-deconvoluted and
    first-order derivative spectra of overlapped band contours
    Web address: https://pubs.acs.org/doi/abs/10.1021/ac00232a034
    """

    def __init__(self, **kwargs):
        """Initialize with Kwargs.
        Parameters
        ----------
        :param: ``kwargs`` : ``dict``
            Arbitrary keyword arguments.
        Args:
        -----
        Keyword arguments. If you do accept ``**kwargs``, make sure
        you link to documentation that describes what keywords are accepted,
        or list the keyword arguments here:
        ``x`` : ``array_like``
            The data to be filtered.  If `x` is not a single or double precision
            floating point array, it will be converted to type `numpy.float64`
            before filtering.
        ``window_length`` : ``int``
            The length of the filter window (i.e. the number of coefficients).
            `window_length` must be a positive odd integer. If `mode` is 'interp',
            `window_length` must be less than or equal to the size of `x`.
        ``polyorder`` : ``int``
            The order of the polynomial used to fit the samples.
            `polyorder` must be less than `window_length`.
    ``deriv`` : ``int, optional``
            The order of the derivative to compute.  This must be a
            nonnegative integer.  The default is 0, which means to filter
            the data without differentiating.
        ``delta`` : ``float, optional``
            The spacing of the samples to which the filter will be applied.
            This is only used if deriv > 0.  Default is 1.0.
        """
        self.__dict__.update(kwargs)

        allowed_keys = ["sav_window", "sav_poly", "sav_deriv"]
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        if not hasattr(self, "delta"):
            self.delta = 1

    def fit(self, x: pd.DataFrame):
        """Fit the transform.
        Parameters
        ----------
        :param: ``x`` : ``pd.DataFrame``
            Dataframe of SPC data.
        Returns
        -------
        :return: ``self``
        """

        # save original data
        self.original_data = x.copy(deep=True)
        self.original_heads = [float(x) for x in list(x)]
        self.original_values = x.values
        self.original_index = x.index.values

        self.result = savgol_filter(
            self.original_values,
            window_length=self.sav_window,
            polyorder=self.sav_poly,
            deriv=self.sav_deriv,
            delta=self.delta,
            axis=1,
        )

        return self

    def transform(self, x: pd.DataFrame):
        """ transform and output dataframe
        Parameters
        ----------
        :param: ``x`` : ``pd.DataFrame``
            Dataframe of SPC data.
        :return: ``pd.DataFrame``
            Return the transformed dataframe.
        """
        return pd.DataFrame(
            self.result, columns=self.original_heads, index=self.original_index
        )