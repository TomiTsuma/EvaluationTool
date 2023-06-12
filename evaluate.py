from sklearn.metrics import *
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow import keras
import os
import csv
import json
# import pyodbc
import joblib
from pandas import pivot_table
from distutils.dir_util import copy_tree
import shutil
import itertools
import pandas as pd
from datetime import date
from PIL import Image
import glob
import numpy as np
from imageio import imsave
from math import log10, floor, ceil
from scipy.stats import linregress
from sklearn.metrics import accuracy_score, f1_score, median_absolute_error, precision_score, recall_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import plotly.figure_factory as ff
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import tensorflow as tf
# from wai.ma.core.matrix import Matrix, helper
# from wai.ma.transformation import SavitzkyGolay2
# from wai.ma.filter import Downsample
from pathlib import Path
print(f'tensorflow version : {tf.version.VERSION}')
# tf.enable_eager_execution()

# ClassifierKwargs = ClassifierKwargs()


def predict_chems(path_to_model, predction_folder_path, chemicals, model_versions, data):
    for model_version in model_versions:

        print(f'Starting prediction using model version {model_version}')
        base_path = Path(path_to_model)
        print(f"This is the path to model {path_to_model}")
        for chemical in chemicals:
            print(chemical)
            preds_comb = pd.DataFrame()
            models_folder = base_path / model_version / chemical / 'std'
            print(f"This is the model path {models_folder}")
            all_models = [x for x in models_folder.glob('**/*.hdf5')]
            print(f"These are the models {all_models}")
            #                             data = pd.read_csv(filename, index_col=[0,1])
            data = data
            new_indices = data.index
            # data.drop(chemical, axis=1, inplace=True)

            for model_path in all_models:

                json_path = model_path.parent.parent / 'model.json'

                with open(json_path) as f:
                    json_ = json.load(f)

                inputs = []

                for i in range(len(json_['Inputs'])):
                    input_name = json_['Inputs'][i]['Name']
#                     print(f'filename: {filename}')
                    train = data.copy(deep=True)

                    for j in range(len(json_['Inputs'][i]['Pre-processing'])):
                        key_ = json_['Inputs'][i]['Pre-processing'][j]['Name']
                        if input_name == 'nir2':
                            input_name = 'nir.2'

                        pickle_path = model_path.parent / 'preprocess' / \
                            f'input.{input_name}.{j}.{key_}.pickle'
                        pickle_ = joblib.load(pickle_path)
                        train = pickle_.fit_transform(train)

                    inputs.append(train.values)

                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(model_path, compile=False)
                preds = pd.DataFrame(model(inputs).numpy())
                preds_comb = pd.concat([preds_comb, preds], axis=1)
            preds_comb = preds_comb.median(axis=1)
            preds_comb.index = new_indices

            os.makedirs(
                f'{predction_folder_path}/{model_version}', exist_ok=True)
            preds_comb.to_csv(
                f'{predction_folder_path}/{model_version}/{chemical}_preds.csv')
        print(f'Finalizing prediction using model version {model_version}')


def subset_data(spc_data, path_to_codes_subset):
    print(f'Subsetting data')
    codes = pd.read_csv(path_to_codes_subset, index_col=0)
    codes = codes.dropna()
    codes = codes.drop_duplicates()
    codes = codes.index
    df = spc_data.reindex(codes)
    df = df.drop_duplicates()
    return df


def join_preds_wet(path_to_wet, output_path, model_versions, chemicals, predction_folder_path):
    os.makedirs(Path(os.path.join(output_path)) /
                'saved_models', exist_ok=True)
    df = pd.read_csv(path_to_wet, index_col=0)
    df.index = df.index.str.strip()
    for chemical in chemicals:
        df_ = df[chemical]
        df_ = df_.to_frame()
        df_ = df_.dropna()
        comb_df = pd.DataFrame()

        for model_version in model_versions:
            print(
                f'Starting to join predictions and wetchem for evaluation for model version {model_version}')
            print(f"Prediction folder path {predction_folder_path}")
            df_preds = pd.read_csv(
                f'{predction_folder_path}/{model_version}/{chemical}_preds.csv', index_col=0, header=None)

            df_preds = df_preds.reindex(df_.index)
            # df_preds = df_preds.dropna()
            print("Preds", df_preds)
            df_preds = df_preds.rename(
                columns={1: f'{model_version}_regression'})
            comb_df = pd.concat([comb_df, df_preds], axis=1)
        print("Preds", df_preds)
        df_wet = df_.rename(columns={chemical: 'y_true_val'})
        print("Wet", df_wet)
        comb_df = pd.concat([comb_df, df_wet], axis=1)
        comb_df = comb_df.dropna()
        comb_df = comb_df[~comb_df.index.duplicated()]
        print(comb_df.shape)

        comb_df.to_csv(Path(os.path.join(output_path, 'saved_models')
                            ) / f'{chemical}_False_y_pred_list_df.csv')
        comb_df.to_pickle(Path(os.path.join(output_path)) / 'saved_models' /
                          f'{chemical}_{model_version}_regression_False_score_trained.pkl')


def join_diff_models_data(output_path, model_versions, chemicals, name_of_subset):

    for chemical in chemicals:
        print(
            f'Starting to join predictions and wetchem for evaluation for different models version for chemical {chemical}')
        comb_df = pd.DataFrame()
        for model_version in model_versions:

            df = pd.read_csv(Path(os.path.join(output_path, model_version)) /
                             f'{chemical}_False_y_pred_list_df.csv', index_col=0)
            y_true_val_series = df['y_true_val']
            comb_df = pd.concat([comb_df, df], axis=1)
            comb_df = comb_df.drop(columns=['y_true_val'])
            comb_df.to_pickle(Path(os.path.join(
                output_path))/f'{name_of_subset}' / 'saved_models' / f'{chemical}_{model_version}_regression_False_score_trained.pkl')

        comb_df = pd.concat([y_true_val_series, comb_df], axis=1)
        os.makedirs(os.path.join(
            output_path, f'{name_of_subset}', 'saved_models'), exist_ok=True)
        comb_df.to_csv(Path(output_path)/f'{name_of_subset}' /
                       'saved_models' / f'{chemical}_False_y_pred_list_df.csv')


def get_spectra(path_to_spectra):
    spc_data = pd.read_csv(path_to_spectra, index_col=0, engine='c')

    return spc_data


class EvaluationTool:

    def __init__(self, **kwargs: dict):
        self.__dict__.update(kwargs)

        allowed_keys = ["chemical", "out_path", "method", "method2"]

        self.__dict__.update((k, v)
                             for k, v in kwargs.items() if k in allowed_keys)

        if not hasattr(self, "chemical"):
            self.chemical = "ph"
        if not hasattr(self, "out_path"):
            self.out_path = os.getcwd()

        if not hasattr(self, 'method'):
            self.method = "regression"

        if not hasattr(self, 'method2'):
            self.method2 = "classification"

    # helper function to create new directory

    @staticmethod
    def _create_dir_if_not_exists(path):
        import pathlib
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    # Helper function to convert results to 3 significant figures
    @staticmethod
    def _round_sig(x, sig=3):
        if x == 0:
            return 0
        else:
            # return round(x, sig-int(ceil(log10(abs(x))))-1)
            return x

    def reversed_chemical_name(self, chemical):

        if chemical == 'ec_salts':
            return 'ec'

        elif chemical == 'exchangeable_acidity':
            return 'exchangeable'

        elif chemical == 'total_nitrogen':
            return 'total'

        elif chemical == 'organic_carbon':
            return 'organic'

        elif chemical == 'reactive_carbon':
            return 'reactive'

        elif chemical == 'reactive_carbon':
            return 'reactive'

        elif chemical == 'phosphorus_olsen':
            return 'olsen'

        else:
            pass

        return chemical

    def correct_chemical_name(self, chemical):

        if chemical == 'ec':
            index_ = 2
            chemical = 'ec_salts'
        elif chemical == 'exchangeable':
            chemical = 'exchangeable_acidity'
            index_ = 2
        elif chemical == 'total':
            chemical = 'total_nitrogen'
            index_ = 2
        elif chemical == 'organic':
            chemical = 'organic_carbon'
            index_ = 2

        elif chemical == 'reactive':

            chemical = 'reactive_carbon'
            index_ = 2

        elif chemical == 'phosphorus':

            chemical = 'phosphorus_olsen'
            index_ = 2

        else:
            index_ = 1

        return chemical, index_

    def get_chem_model_map(self, training_pred_score_path, chemical):
        models = []
        original_path = os.path.join(training_pred_score_path, 'saved_models/')
        # grab all lgb and xgb models since they have different extentions
        delimeter = "/*_regression_False_score_trained.pkl"
        path1 = glob.glob(original_path + delimeter)

        for path in path1:
            if chemical in os.path.basename(path):
                path_base = os.path.basename(path)
                model = path_base.split('_')[-5]
                models.append(model)
        chem_model_map = {chemical: models}

        return chem_model_map

    def _wetchem_statistics(self, df):

        df2 = pd.DataFrame(columns=['no_samples', 'mean', 'median', 'sd', 'minimum',
                           'maximum', 'quantile_25', 'quantile_75', 'kurtosis', 'skew'], index=df.columns)
        for chem in df.columns:

            sd = df[chem].std()
            minimum = df[chem].min()
            kurtosis = df[chem].kurtosis()
            skew = df[chem].skew()
            maximum = df[chem].max()
            mean = df[chem].mean()
            median = df[chem].median()
            quantile_25 = df[chem].quantile(q=0.25)
            quantile_75 = df[chem].quantile(q=0.75)
            no_samples = len(df[chem].dropna())

            df2.at[chem, 'no_samples'] = no_samples
            if chem.endswith("regression"):
                pass
            else:
                df2.at[chem, 'minimum'] = self._round_sig(minimum)
                df2.at[chem, 'maximum'] = self._round_sig(maximum)
                df2.at[chem, 'quantile_25'] = self._round_sig(quantile_25)
                df2.at[chem, 'quantile_75'] = self._round_sig(quantile_75)
                df2.at[chem, 'kurtosis'] = self._round_sig(kurtosis)
                df2.at[chem, 'skew'] = self._round_sig(skew)
                df2.at[chem, 'mean'] = self._round_sig(mean)
                df2.at[chem, 'median'] = self._round_sig(median)
                df2.at[chem, 'sd'] = self._round_sig(sd)

        return df2

    def ppm_to_percentage(self, cec_df, training_pred_score_path, df_perc, df, num_divider, chemical):
        cec_models_trained = []
        chemical_models_trained = []

        for col in cec_df:
            if col != 'y_true_val':
                cec_df.rename(columns={col: f'{col}_cec'}, inplace=True)
                cec_models_trained.append(f'{col}_cec')

        for col in df:
            if col != 'y_true_val':
                chemical_models_trained.append(col)

        for col in df.columns:
            if col == 'y_true_val':

                df_perc['y_true_val'] = (
                    (df[col] / num_divider) / cec_df[col]) * 100

            else:
                pass
        cec_chemical_models_list = cec_models_trained + chemical_models_trained
        combined_list = list(itertools.combinations(
            cec_chemical_models_list, 2))
        combined_list = [list(i) for i in combined_list]  # list of lists

        for item in enumerate(combined_list):
            if item[1][0].endswith('cec') and item[1][1].endswith('cec'):
                combined_list.pop(item[0])

            elif item[1][0].endswith('cec') or item[1][1].endswith('cec'):
                pass
            else:
                combined_list.pop(item[0])
#         combined_list.remove(combined_list[0])
        print(combined_list)
#         combined_list.remove(combined_list[-1])

        writer = pd.ExcelWriter(os.path.join(
            training_pred_score_path, 'saved_models', f'{chemical}_%_False_y_pred_list_df.xlsx'))

        cec_list = []
        other_chem_list = []
        for list_ in combined_list:
            other_chem_list.append(list_[1])
            cec_list.append(list_[0])

        per_model_data_dict = {}
        for i in range(len(cec_list)):

            per_model_data = (
                (df[other_chem_list[i]] / num_divider) / cec_df[cec_list[i]]) * 100
            per_model_data_dict.update({other_chem_list[i]: per_model_data})
            df_perc[other_chem_list[i]] = per_model_data_dict.get(
                other_chem_list[i])

        df_perc.to_excel(writer, sheet_name=f"{cec_list[0]}")
        writer.save()

        chem_model_map = self.get_chem_model_map(
            training_pred_score_path, chemical)
        model_names = chem_model_map.get(chemical)

        model_names = chem_model_map.get(chemical)

        df_ = pd.read_excel(os.path.join(training_pred_score_path, 'saved_models',
                            f'{chemical}_%_False_y_pred_list_df.xlsx'), index_col=0, engine='openpyxl')

        df_ = df_.dropna()
        df_.to_excel(os.path.join(training_pred_score_path,
                     'saved_models', f'{chemical}_%_False_y_pred_list_df.xlsx'))
        df_.to_csv(os.path.join(training_pred_score_path,
                   'saved_models', f'{chemical}_%_False_y_pred_list_df.csv'))
        for model_name in model_names:
            df.to_pickle(os.path.join(training_pred_score_path, 'saved_models',
                         f'{chemical}_%_{model_name}_regression_False_score_trained.pkl'))
        return

    def _wetchem_phosphorus_df(self, df):
        df_phosphorus = pd.DataFrame(columns=[
                                     'no_samples', 'recall_score', 'precision_score', 'f1_score', 'accuracy_score'], index=df.columns)
        return df_phosphorus

    def _preds_vs_wet_statistics(self, training_pred_score_path, chemical, codes=None):
        print("....Predictions vs Wetchem statistics..........")
        added_chemicals = ['calcium_%', 'potassium_%', 'magnesium_%']
#         if chemical =='calcium':
#             chemical = 'calcium_%'
#         elif chemical =='potassium':
#             chemical = 'potassium_%'
#         else:
#             pass
        method = 'regression'
        method2 = self.method2

        print("Method 1", method)
        print("Method 2", method2)
        _wetchem_statistics = self._wetchem_statistics
        _wetchem_phosphorus_df = self._wetchem_phosphorus_df

        if codes != None:
            df = pd.read_csv(os.path.join(training_pred_score_path, 'saved_models/',
                             f'{chemical}_False_y_pred_list_df.csv'), index_col=0)
            df = df.reindex(codes)
        else:
            #             if chemical in added_chemicals:
            #                 df = pd.read_excel(os.path.join(training_pred_score_path , 'saved_models/',f'{chemical}_False_y_pred_list_df.xlsx'),index_col=0)
            #             else:
            df = pd.read_csv(os.path.join(training_pred_score_path, 'saved_models/',
                             f'{chemical}_False_y_pred_list_df.csv'), index_col=0)
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", df)
        comb_df_list = []
        comb_df_list2 = []

        chem_model_map = self.get_chem_model_map(
            training_pred_score_path, chemical)
        print("Chemical models map", chem_model_map)
        for models_ in chem_model_map.get(chemical):
            model_first_name = models_

#                 df = pd.read_csv(os.path.join(training_pred_score_path , 'saved_models/',f'{chemical}_False_y_pred_list_df.csv'),index_col=0)
            print(
                "CV Model Evaluation for chemical={}, mlmodel={}".format(
                    chemical, model_first_name
                )
            )
            df = df.dropna()

            df_Q0_Q025 = df[df.y_true_val.between(
                *df.y_true_val.quantile([0, 0.25]).tolist())]
            df_Q025_Q050 = df[df.y_true_val.between(
                *df.y_true_val.quantile([0.25, 0.5]).tolist())]
            df_Q050_Q075 = df[df.y_true_val.between(
                *df.y_true_val.quantile([0.5, 0.75]).tolist())]
            df_Q075_Q1 = df[df.y_true_val.between(
                *df.y_true_val.quantile([0.75, 1]).tolist())]
            df_Q0_Q050 = df[df.y_true_val.between(
                *df.y_true_val.quantile([0, 0.5]).tolist())]
            df_Q050_Q075 = df[df.y_true_val.between(
                *df.y_true_val.quantile([0.5, 0.75]).tolist())]
            xq1 = df_Q0_Q025['y_true_val']
            yq1 = df_Q0_Q025[f'{model_first_name}_regression']
            xq2 = df_Q025_Q050['y_true_val']
            yq2 = df_Q025_Q050[f'{model_first_name}_regression']
            xq3 = df_Q050_Q075['y_true_val']
            yq3 = df_Q050_Q075[f'{model_first_name}_regression']
            xq4 = df_Q075_Q1['y_true_val']
            yq4 = df_Q075_Q1[f'{model_first_name}_regression']

            x = df['y_true_val']
            y = df[f'{model_first_name}_regression']
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            df_regression = _wetchem_statistics(
                df[['y_true_val', f'{model_first_name}_regression']])
            df_regression['Model'] = np.nan
            rmse = mean_squared_error(x, y) ** 0.5
            r2_squared = r2_score(x, y)
            rsc = x.std()/rmse
            df_regression.at['y_true_val', 'slope'] = self._round_sig(slope)
            df_regression.at['y_true_val',
                             'intercept'] = self._round_sig(intercept)
            df_regression.at['y_true_val', 'RMSE'] = self._round_sig(rmse)
            df_regression.at['y_true_val', 'RSC'] = self._round_sig(rsc)
            df_regression.at['y_true_val', 'R2'] = self._round_sig(r2_squared)
            df_regression.at['y_true_val', 'RMSECVQ1'] = self._round_sig(
                mean_squared_error(xq1, yq1) ** 0.5)
            df_regression.at['y_true_val', 'RMSECVQ2'] = self._round_sig(
                mean_squared_error(xq2, yq2) ** 0.5)
            df_regression.at['y_true_val', 'RMSECVQ3'] = self._round_sig(
                mean_squared_error(xq3, yq3) ** 0.5)
            df_regression.at['y_true_val', 'RMSECVQ4'] = self._round_sig(
                mean_squared_error(xq4, yq4) ** 0.5)
            df_regression['Model'].fillna(
                f'{model_first_name}_regression', inplace=True)
            df_regression.drop(
                [f'{model_first_name}_regression'], inplace=True)
            df_regression.rename(index={'y_true_val': chemical}, inplace=True)

            comb_df_list.append(df_regression)

        regressor_chemicals = len(comb_df_list)
        if regressor_chemicals > 0:
            comb_df = pd.concat(comb_df_list)
            comb_df.drop_duplicates(inplace=True)
        else:
            comb_df = pd.DataFrame()

        class_chemicals = len(comb_df_list2)
        if class_chemicals > 0:
            comb_df_class = pd.concat(comb_df_list2)
            comb_df_class.drop_duplicates(inplace=True)
        else:
            comb_df_class = pd.DataFrame()

        return comb_df, comb_df_class

    def chem_to_percentage_conversion(self, training_pred_score_path, chemicals_conv, codes=None):
        #
        chems = ['potassium', 'calcium', 'magnesium']

        for chemical in chemicals_conv:
            if chemical not in chems:
                pass
            else:

                #             try:
                df = pd.read_csv(os.path.join(training_pred_score_path, 'saved_models/',
                                 f'{chemical}_False_y_pred_list_df.csv'), index_col=0)
    #             print(df)
    #

                if codes != None:

                    df = pd.read_csv(os.path.join(
                        training_pred_score_path, 'saved_models/', f'{chemical}_False_y_pred_list_df.csv'), index_col=0)

                    cec_df = pd.read_csv(os.path.join(
                        training_pred_score_path, 'saved_models/', 'cec_False_y_pred_list_df.csv'), index_col=0)
                    df = df.reindex(codes)

                    cec_df = cec_df.reindex(codes)
                else:
                    df = pd.read_csv(os.path.join(
                        training_pred_score_path, 'saved_models/', f'{chemical}_False_y_pred_list_df.csv'), index_col=0)

                    cec_df = pd.read_csv(os.path.join(
                        training_pred_score_path, 'saved_models/', 'cec_False_y_pred_list_df.csv'), index_col=0)

                df_perc = pd.DataFrame(
                    np.zeros((df.shape[0], df.shape[1])), index=df.index, columns=df.columns)
                if chemical == 'calcium':
                    num_divider = 200
                    self.ppm_to_percentage(
                        cec_df, training_pred_score_path, df_perc, df, num_divider, chemical)
                elif chemical == 'magnesium':
                    num_divider = 120
                    self.ppm_to_percentage(
                        cec_df, training_pred_score_path, df_perc, df, num_divider, chemical)
                else:
                    num_divider = 390
                    self.ppm_to_percentage(
                        cec_df, training_pred_score_path, df_perc, df, num_divider, chemical)
#             except:
#                 pass
        return


class PlotModelStats:

    """
    This is a class where various plots are produced to display models performance
    """

    def __init__(self, **kwargs: dict):

        self.__dict__.update(kwargs)
        allowed_keys = ['outpath', 'method', 'chemicals_with_percentage']

        self.__dict__.update((k, v)
                             for k, v in kwargs.items() if k in allowed_keys)

        if not hasattr(self, 'outpath'):
            self.outhpath = os.getcwd()

        if not hasattr(self, 'method'):
            self.method = "regression"
        if not hasattr(self, 'chemicals_with_percentage'):
            self.method = "chemicals_with_percentage"

    def _plot_confusion_matrix_plotly(self, cm, training_pred_score_path, chemical, model_first_name, area, normalize=False, plot=False, codes=None):
        cm_path = training_pred_score_path

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = cm.round(4)
            normalized = 'original_normalized'
        else:
            normalized = 'non_normalized'
        target_names = ['very_low', 'low', 'optimum', 'high']
        fig = ff.create_annotated_heatmap(cm.transpose(
        ), x=target_names, y=target_names, colorscale='blues', showscale=True, reversescale=False)
        fig['layout']['xaxis'].update(side='bottom', title='True')
        fig['layout']['yaxis'].update(side='left', title='Predicted')
        fig.update_layout(
            title=f'{area} {model_first_name} {chemical} confusion matrix')
        if plot == True:

            fig.show()
        else:

            cm_plot_path_subset = os.path.join(cm_path, 'saved_models', 'confusion_matrix_subset', '{}_{}_{}_cm.png'.format(
                chemical, model_first_name, normalized))
            cm_plot_path = os.path.join(cm_path, 'saved_models', 'confusion_matrix', '{}_{}_{}_cm.png'.format(
                chemical, model_first_name, normalized))
            if codes != None:

                os.makedirs(os.path.join(cm_path, 'saved_models',
                            'confusion_matrix_subset'), exist_ok=True)
                fig.write_image(cm_plot_path_subset)
            else:
                os.makedirs(os.path.join(cm_path, 'saved_models',
                            'confusion_matrix'), exist_ok=True)
                fig.write_image(cm_plot_path)

    def _create_confusion_matrices(self, training_pred_score_path, region, chem_correction, chemicals_conv, codes=None):
        area = region
        method = 'classification'

        path_to_saved_models = os.path.join(
            training_pred_score_path, 'saved_models')
#
        guides = {
            'boron': [-1e6, 0.5, 0.8, 1, 1e6],
            'calcium_%': [-1e6, 40, 60, 65, 1e6],
            'cec': [-1e6, 8, 15, 20, 1e6],
            'copper': [-1e6, 1, 1.5, 8, 1e6],
            'iron': [-1e6, 20, 30, 50, 1e6],
            'magnesium_%': [-1e6, 8, 10, 15, 1e6],
            'manganese': [-1e6, 10, 20, 100, 1e6],
            'potassium_%': [-1e6, 1.5, 3, 5, 1e6],
            'ph': [-1e6, 5.5, 5.8, 6.4, 1e6],
            'organic_carbon': [-1e6, 1, 2, 4, 1e6],
            'sulphur': [-1e6, 5, 10, 20, 1e6],
            'total_nitrogen': [-1e6, 0.1, 0.2, 0.25, 1e6],
            'zinc': [-1e6, 1, 2, 4, 1e6],
            'phosphorus': [-1e6, 10, 30, 50, 1e6]
        }

        chem_with_plots = []
        for chemical in chemicals_conv:

            if method == 'classification':
                chem_model_map = EvaluationTool().get_chem_model_map(
                    training_pred_score_path, chemical)
                print(chem_model_map)
                for models_ in chem_model_map.get(chemical):
                    model_first_name = models_
                    if chemical in guides.keys():

                        chem_with_plots.append(chemical)
                        df = pd.read_csv(os.path.join(
                            training_pred_score_path, 'saved_models/{}_False_y_pred_list_df.csv'.format(chemical)), index_col=0)
                        y_val = df['y_true_val'].values
                        y_preds = df[f'{model_first_name}_regression'].values
                        preds_vs_wet_temp = pd.DataFrame(
                            [y_val.reshape(-1,), y_preds.reshape(-1,)], index=['wet', 'preds']).T
                    #     break
                        preds_vs_wet_temp[f'{chemical}_labels'] = pd.cut(preds_vs_wet_temp['wet'].dropna(
                        ), bins=guides[f'{chemical}'], labels=["very_low", "low", "optimum", "high"])
                        preds_vs_wet_temp["label_code"] = preds_vs_wet_temp[
                            f'{chemical}_labels'].cat.codes

                        preds_vs_wet_temp[f'{chemical}_labels_preds'] = pd.cut(preds_vs_wet_temp['preds'].dropna(
                        ), bins=guides[f'{chemical}'], labels=["very_low", "low", "optimum", "high"])

                        if chem_correction == True and chemical == 'phosphorus':
                            preds_vs_wet_temp = self.p_correction_v3(
                                preds_vs_wet_temp, chemical)

                            preds_vs_wet_temp["label_preds_code"] = preds_vs_wet_temp[
                                f'{chemical}_labels_preds'].cat.codes
#
                        else:
                            #                             print(f'{chemical}.....TRUEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')

                            preds_vs_wet_temp["label_preds_code"] = preds_vs_wet_temp[
                                f'{chemical}_labels_preds'].cat.codes

                        target_names = ['very_low', 'low', 'optimum', 'high']
                        if codes != None:
                            preds_vs_wet_temp.to_csv(os.path.join(
                                training_pred_score_path, f'{chemical}_{model_first_name}_classes_subset.csv'))
                        else:
                            preds_vs_wet_temp.to_csv(os.path.join(
                                training_pred_score_path, f'{chemical}_{model_first_name}_classes.csv'))
                        # cm = confusion_matrix(preds_vs_wet_temp[f'{chemical}_labels'], preds_vs_wet_temp[f'{chemical}_labels_preds'], labels=target_names)

                        cm = confusion_matrix(
                            preds_vs_wet_temp['label_code'], preds_vs_wet_temp['label_preds_code'], labels=[0, 1, 2, 3])

                        if codes != None:
                            self._plot_confusion_matrix_plotly(
                                cm, training_pred_score_path, chemical, model_first_name, area, normalize=False, plot=False, codes=codes)
                            self._plot_confusion_matrix_plotly(
                                cm, training_pred_score_path, chemical, model_first_name, area, normalize=True, plot=False, codes=codes)
                            self.combine_cm_matrices(
                                training_pred_score_path, codes=codes)
                        else:
                            self._plot_confusion_matrix_plotly(
                                cm, training_pred_score_path, chemical, model_first_name, area, normalize=False, plot=False, codes=None)
                            self._plot_confusion_matrix_plotly(
                                cm, training_pred_score_path, chemical, model_first_name, area, normalize=True, plot=False, codes=None)
                            self.combine_cm_matrices(
                                training_pred_score_path, codes=None)

        # to handle duplicate chemicals in list
        chem_with_list = OrderedDict.fromkeys(chem_with_plots)
        for chemical in chem_with_list:
            if method == 'regression':
                #                 chemical = EvaluationTool().reversed_chemical_name(chemical)
                if codes != None:

                    self.combine_confusion_matrices(
                        training_pred_score_path, chemical, codes=codes)
                else:
                    self.combine_confusion_matrices(
                        training_pred_score_path, chemical, codes=None)
        return

    def combine_cm_matrices(self, training_pred_score_path, codes=None):
        # TO DO : change from hardcoded no of images and dimensions to dynamic
        saved_models_path = training_pred_score_path
        if codes != None:
            delimiter_subset = '/saved_models/confusion_matrix_subset/*original_normalized_cm.png'
        else:
            delimiter = '/saved_models/confusion_matrix/*original_normalized_cm.png'
        img = np.zeros([1920, 2560, 3], dtype=np.uint8)
        img.fill(255)
        if codes != None:
            os.makedirs(os.path.join(saved_models_path, 'saved_models',
                        'confusion_matrix_subset'), exist_ok=True)
            cm_blank_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix_subset', 'norm_cm_blank.png')
            cm_comb_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix_subset', 'comb_norm_cm.png')
            imsave(cm_blank_path, img)
        else:
            os.makedirs(os.path.join(saved_models_path,
                        'saved_models', 'confusion_matrix'), exist_ok=True)
            cm_blank_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix', 'norm_cm_blank.png')
            cm_comb_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix', 'comb_norm_cm.png')
            imsave(cm_blank_path, img)
        if codes != None:
            all_files = sorted(glob.glob(saved_models_path + delimiter_subset))
        else:
            all_files = sorted(glob.glob(saved_models_path + delimiter))
        blank_image = Image.open(cm_blank_path)
        height = 0
        width = 0
        for file in all_files:
            cm = Image.open(file)
            blank_image.paste(cm, (width, height))
            if width < 1920:
                width += 640
            else:
                width = 0
                height += 480
        blank_image.save(cm_comb_path)
        if codes != None:
            delimiter = '/saved_models/confusion_matrix_subset/*_non_normalized_cm.png'
        else:
            delimiter = '/saved_models/confusion_matrix/*_non_normalized_cm.png'
        img = np.zeros([1920, 2560, 3], dtype=np.uint8)
        img.fill(255)
        if codes != None:
            cm_blank_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix_subset', 'unnorm_cm_blank.png')
            cm_comb_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix_subset', 'comb_unnorm_cm.png')
        else:
            cm_blank_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix', 'unnorm_cm_blank.png')
            cm_comb_path = os.path.join(
                saved_models_path, 'saved_models', 'confusion_matrix', 'comb_unnorm_cm.png')
        imsave(cm_blank_path, img)
        if codes != None:
            all_files = sorted(glob.glob(saved_models_path + delimiter_subset))
        else:
            all_files = sorted(glob.glob(saved_models_path + delimiter))
        blank_image = Image.open(cm_blank_path)
        height = 0
        width = 0
        for file in all_files:
            cm = Image.open(file)
            blank_image.paste(cm, (width, height))
            if width < 1920:
                width += 640
            else:
                width = 0
                height += 480
        blank_image.save(cm_comb_path)
        return cm_comb_path

    def _PlotScatter(self, training_pred_score_path, chemicals_conv, plot=True, codes=None):

        method = "regression"

        path_to_saved_models = os.path.join(
            training_pred_score_path, 'saved_models')
        os.makedirs(os.path.join(path_to_saved_models,
                    'scatter_plots'), exist_ok=True)

        chemicals_conv = list(dict.fromkeys(chemicals_conv))

        for chemical in chemicals_conv:

            print(chemical)
            if method == 'regression':
                print('yes')
                chem_model_map = EvaluationTool().get_chem_model_map(
                    training_pred_score_path, chemical)
                print(chem_model_map)
                for models_ in chem_model_map.get(chemical):
                    model_first_name = models_
                    print(model_first_name)

                    if codes != None:
                        if chemical != 'calcium_%' or chemical != 'potassium_%' or chemical != 'magnesium_%':
                            df = pd.read_csv(os.path.join(
                                training_pred_score_path, 'saved_models/{}_False_y_pred_list_df.csv'.format(chemical)), index_col=0)
                        else:
                            df = pd.read_excel(os.path.join(
                                training_pred_score_path, 'saved_models/{}_False_y_pred_list_df.xlsx'.format(chemical)), index_col=0, engine='openpyxl')
                        df.index = df.index.str.strip()
                        df = df.reindex(codes)
                    else:
                        if chemical != 'calcium_%' or chemical != 'potassium_%' or chemical != 'magnesium_%':
                            df = pd.read_csv(os.path.join(
                                training_pred_score_path, 'saved_models/{}_False_y_pred_list_df.csv'.format(chemical)), index_col=0)
                        else:
                            df = pd.read_excel(os.path.join(
                                training_pred_score_path, 'saved_models/{}_False_y_pred_list_df.xlsx'.format(chemical)), index_col=0, engine='openpyxl')
                        df.index = df.index.str.strip()
                        print(df.head())
                        print('yes')
                    df = df[[f'{model_first_name}_regression', 'y_true_val']]
                    y_max = df[f'{model_first_name}_regression'].max()
                    x_max = df['y_true_val'].max()
                    y_min = df[f'{model_first_name}_regression'].min()
                    x_min = df['y_true_val'].min()
                    max_dim = max(x_max, y_max)
                    fig = px.scatter(df, x='y_true_val', y=f'{model_first_name}_regression',
                                     trendline="ols")
                    fig.add_trace(
                        go.Scatter(
                            x=[0, x_max + x_max/(x_max/2)],
                            y=[0, x_max + x_max/(x_max/2)],
                            mode="lines",
                            line=go.scatter.Line(color="gray"),
                            name='1:1 Line',
                            showlegend=True)
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 0],
                            y=[0, 0],
                            mode="lines",
                            line=go.scatter.Line(color="blue"),
                            name='Linear model line',
                            showlegend=True)
                    )

                    if chemical == 'ph':
                        fig.update_layout(title=f'{model_first_name} Predictions vs Wetchem Scatter for {chemical}', xaxis=dict(range=(df['y_true_val'].min(
                        ) - 1, max_dim + 1), constrain='domain', title='wetchem'), yaxis=dict(range=(df['y_true_val'].min() - 1, max_dim + 1), constrain='domain', title='predictions'))
                    elif chemical == 'sand':
                        fig.update_layout(title=f'{model_first_name} Predictions vs Wetchem Scatter for {chemical}', xaxis=dict(range=(df['y_true_val'].min(
                        ) - 10, max_dim + 10), constrain='domain', title='wetchem'), yaxis=dict(range=(df['y_true_val'].min() - 10, max_dim + 10), constrain='domain', title='predictions'))
                    else:
                        if y_min < 0:
                            fig.update_layout(title=f'{model_first_name} Predictions vs Wetchem Scatter for {chemical}', xaxis=dict(range=(0, max_dim + df['y_true_val'].median(
                            )/0.75), constrain='domain', title='wetchem'), yaxis=dict(range=(y_min, max_dim + df['y_true_val'].median()/0.75), constrain='domain', title='predictions'))
                        else:

                            fig.update_layout(title=f'{model_first_name} Predictions vs Wetchem Scatter for {chemical}', xaxis=dict(range=(0, max_dim + df['y_true_val'].median(
                            )/0.75), constrain='domain', title='wetchem'), yaxis=dict(range=(0, max_dim + df['y_true_val'].median()/0.75), constrain='domain', title='predictions'))
                    if plot:

                        fig.show()
                    else:
                        if codes != None:

                            cm_plot_path = os.path.join(
                                training_pred_score_path, 'saved_models', 'scatter_plots_subset', '{}_{}.png'.format(chemical, model_first_name))
                            os.makedirs(os.path.join(
                                training_pred_score_path, 'saved_models', 'scatter_plots_subset'), exist_ok=True)
                        else:
                            cm_plot_path = os.path.join(
                                training_pred_score_path, 'saved_models', 'scatter_plots', '{}_{}.png'.format(chemical, model_first_name))
                            os.makedirs(os.path.join(
                                training_pred_score_path, 'saved_models', 'scatter_plots'), exist_ok=True)
                        fig.write_image(cm_plot_path)

        for chemical in chemicals_conv:
            #             chemical = EvaluationTool().reversed_chemical_name(chemical)
            if method == 'regression':
                if codes != None:

                    self.combine_scatter_plots(
                        training_pred_score_path, chemical, codes=codes)
                else:
                    if len(models_) > 1:
                        print(
                            "REacheed chem + +++++++++++++++++++++++++++++++++ ++ ", chemical)
                        self.combine_scatter_plots(
                            training_pred_score_path, chemical, codes=None)
                    else:
                        pass

        return

    def Image_combiner(self, all_files):
        images = [Image.open(x) for x in all_files]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:

            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im

    def combine_scatter_plots(self, training_pred_score_path, chemical, codes=None):
        chemical_to_separate = ['calcium',
                                'magnesium', 'potassium', 'phosphorus']
        chem_map = EvaluationTool().get_chem_model_map(
            training_pred_score_path, chemical)
        if codes != None:

            delimiter = f'/saved_models/scatter_plots_subset/{chemical}*'
        else:
            delimiter = f'/saved_models/scatter_plots/{chemical}*'

        if chemical == 'ph':

            delimiter = f'/saved_models/scatter_plots/{chemical}_*'
            all_files = sorted(glob.glob(training_pred_score_path + delimiter))

        elif chemical in chemical_to_separate:
            delimiter = f'/saved_models/scatter_plots/{chemical}_DL*'
            all_files = sorted(glob.glob(training_pred_score_path + delimiter))

        else:

            all_files = sorted(glob.glob(training_pred_score_path + delimiter))

        new_im = self.Image_combiner(all_files)
        if codes != None:

            new_im.save(os.path.join(training_pred_score_path, 'saved_models',
                        'scatter_plots_subset', f'{chemical} combined_scatter_plot.png'))
        else:
            new_im.save(os.path.join(training_pred_score_path, 'saved_models',
                        'scatter_plots', f'{chemical} combined_scatter_plot.png'))

    def combine_confusion_matrices(self, training_pred_score_path, chemical, codes=None):
        chem_map = EvaluationTool().get_chem_model_map(
            training_pred_score_path, chemical)
        if codes != None:
            chem_delimiter = f'/saved_models/confusion_matrix_subset/{chemical}*'
        else:
            chem_delimiter = f'/saved_models/confusion_matrix/{chemical}*'
        all_files_chem = sorted(
            glob.glob(training_pred_score_path + chem_delimiter))
        if chemical == 'ph':
            number_of_plots = len(chem_map[chemical])
            all_files_chem = all_files_chem[:number_of_plots]
        else:
            pass
        non_norm_list = []
        norm_list = []
        for path in all_files_chem:
            if path.endswith('_non_normalized_cm.png'):
                non_norm_list.append(path)
            else:
                norm_list.append(path)

        new_im_norm = self.Image_combiner(norm_list)
        new_im_un_norm = self.Image_combiner(non_norm_list)
        if codes != None:

            new_im_norm.save(os.path.join(training_pred_score_path, 'saved_models',
                             'confusion_matrix_subset', f'{chemical} combined_normalized_confusion.png'))
            new_im_un_norm.save(os.path.join(training_pred_score_path, 'saved_models',
                                'confusion_matrix_subset', f'{chemical} combined_unormalized_confusion.png'))
        else:
            new_im_norm.save(os.path.join(training_pred_score_path, 'saved_models',
                             'confusion_matrix', f'{chemical} combined_normalized_confusion.png'))
            new_im_un_norm.save(os.path.join(training_pred_score_path, 'saved_models',
                                'confusion_matrix', f'{chemical} combined_unormalized_confusion.png'))

    def p_correction_v3(self, preds_vs_wet_temp, chemical):
        guides = {
            'phosphorus': [-1e6, 30, 50, 80, 1e6], }

        preds_vs_wet_temp[f'{chemical}_labels_preds'] = pd.cut(preds_vs_wet_temp['preds'].dropna(
        ), bins=guides[f'{chemical}'], labels=["very_low", "low", "optimum", "high"])
        preds_vs_wet_temp["label_preds_code"] = preds_vs_wet_temp[
            f'{chemical}_labels_preds'].cat.codes

        return preds_vs_wet_temp


def saved_eval_params(training_pred_score_path, project_name, lines, lines_dict):
    import logging
    today = date.today()

    d1 = today.strftime("%Y/%m/%d")
    d1 = d1.replace('/', '_')
    dl_new = f'{d1}_{project_name}_v2.0_v2.2_v5'
    LOG_FILENAME = os.path.join(
        training_pred_score_path, f'reproducing_evaluation.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    for num, line in enumerate(lines):

        logging.info(f'{lines_dict[num]}: {line}')


def delete_uneccessay_files(training_pred_score_path):

    path_list = []
    for path, subdirs, files in os.walk(training_pred_score_path):
        for name in files:
            print(os.path.join(path, name))
            path_ = os.path.join(path, name)
            if 'norm_cm_blank' in path_:

                path_list.append(path_)

            elif 'classes_Evaluation' in path_:

                path_list.append(path_)

    return [os.remove(x) for x in path_list]


def delete_files(training_pred_score_path):

    path_list = []
    for path, subdirs, files in os.walk(training_pred_score_path):
        for name in files:
            print(os.path.join(path, name))
            path_ = os.path.join(path, name)
            if '.png' in path_:

                path_list.append(path_)

    return [os.remove(x) for x in path_list]


class Models_Summary:

    """
    This is a class where all evaluation reports plots are produced for Models Trained
    """

    def __init__(self, **kwargs: dict):

        self.__dict__.update(kwargs)
        allowed_keys = ['outpath']

        self.__dict__.update((k, v)
                             for k, v in kwargs.items() if k in allowed_keys)

        if not hasattr(self, 'outpath'):

            self.outhpath = os.getcwd()

    def __init__(self, **kwargs: dict):

        self.__dict__.update(kwargs)
        allowed_keys = ['outpath', 'method', 'method2']

        self.__dict__.update((k, v)
                             for k, v in kwargs.items() if k in allowed_keys)

        if not hasattr(self, 'outpath'):

            self.outhpath = os.getcwd()

        if not hasattr(self, 'method'):

            self.method = 'regression'
        if not hasattr(self, 'method2'):

            self.method2 = 'classification'

    def Models_Summary(self, training_pred_score_path, project_name, working_metrics, corrected_chems=None, chem_correction=None, training_pred_score_paths=None, wet_chem_path=None, predict=False, codes=None):

        project_name = "".join(project_name.split("_"))
        added_chemicals = ['calcium_%', 'potassium_%', 'magnesium_%']

        chemicals_conv = []

        path_to_saved_models = os.path.join(
            training_pred_score_path, 'saved_models')
        path_to_saved_models = Path(path_to_saved_models)
        chemicals_conv = [x.name.split("_")[0] for x in path_to_saved_models.glob(
            '**/*False_y_pred_list_df.csv')]

        if wet_chem_path != None:
            wet_chem = pd.read_csv(wet_chem_path, index_col=0)
        method = 'regression'
        method2 = 'classification'
        region = project_name

        if predict == True:
            all_preds = glob.glob(training_pred_score_path + f'/*.csv')
            df_chems = pd.read_csv(all_preds[0], index_col=0)
            chems = pd.read_csv(all_preds[0], index_col=0).columns.tolist()
            os.makedirs(os.path.join(training_pred_score_path,
                        'saved_models'), exist_ok=True)
            shutil.copy2(all_preds[0], os.path.join(
                training_pred_score_path, 'saved_models'))
            predictions_path = glob.glob(os.path.join(
                training_pred_score_path, 'saved_models') + f'/*.csv')

            for chem in chems:

                best_model = best_models.at[chem, 'model']
                df_ = df_chems[chem].to_frame()
                df_wet = wet_chem[chem].to_frame()
                df_ = df_.rename(columns={chem: f'{best_model}_regression'})
                df_wet = df_wet.rename(columns={chem: 'y_true_val'})
                comb_df = pd.concat([df_, df_wet], axis=1)
                comb_df.to_csv(os.path.join(
                    training_pred_score_path, 'saved_models', f'{chem}_False_y_pred_list_df.csv'))
                comb_df.to_pickle(os.path.join(training_pred_score_path, 'saved_models',
                                  f'{chem}_{best_model}_regression_False_score_trained.pkl'))

        else:
            pass

        if codes != None:
            #             EvaluationTool().chem_to_percentage_conversion(training_pred_score_path, codes=codes)
            PlotModelStats()._PlotScatter(training_pred_score_path,
                                          chemicals_conv, plot=False, codes=codes)
            try:
                PlotModelStats()._create_confusion_matrices(
                    training_pred_score_path, region, chemicals_conv, codes=codes)
            except:
                pass
            path_to_save = os.path.join(
                training_pred_score_path, 'saved_models', 'Predictions_subset')
            os.makedirs(path_to_save, exist_ok=True)
        else:
            if chem_correction:
                #                 EvaluationTool().chem_to_percentage_conversion(training_pred_score_path,chem_correction)
                #                 path_to_remove = [x for x in os.listdir(os.path.join(training_pred_score_path, 'saved_models') ) if 'cec' in x]
                #                 try:
                #                     for ptr in path_to_remove:
                #                          os.remove(os.path.join(training_pred_score_path,'saved_models/',ptr))
                #                     chemicals_conv.remove('cec')
                #                 except:
                #                     pass

                PlotModelStats()._PlotScatter(training_pred_score_path, chemicals_conv, plot=False)
                PlotModelStats()._create_confusion_matrices(training_pred_score_path, region,
                                                            chem_correction, chemicals_conv=chemicals_conv, codes=codes)

            else:

                #                 pass
                EvaluationTool().chem_to_percentage_conversion(
                    training_pred_score_path, chemicals_conv)
                PlotModelStats()._PlotScatter(training_pred_score_path, chemicals_conv, plot=False)
                PlotModelStats()._create_confusion_matrices(training_pred_score_path, region,
                                                            chem_correction, chemicals_conv=chemicals_conv, codes=codes)

            path_to_save = os.path.join(
                training_pred_score_path, 'saved_models', 'Predictions')
            os.makedirs(path_to_save, exist_ok=True)

        p_df = pd.DataFrame()
        other_chems = pd.DataFrame()
        add_chems = ['calcium_%', 'magnesium_%', 'potassium_%']
        if chem_correction == True:
            chemicals_conv = ['zinc', 'phosphorus', 'potassium']
        for chemical in chemicals_conv:
            if chemical in add_chems:
                print("Chemicals was in add_chems")
                pass

            if codes != None:
                print()
                df = pd.read_csv(os.path.join(training_pred_score_path, 'saved_models',
                                 f'{chemical}_False_y_pred_list_df.csv'), index_col=0)
                df = df.reindex(codes)
                df.to_csv(os.path.join(training_pred_score_path, 'saved_models',
                          'Predictions_subset', f'predictions_{chemical}.csv'))

            else:
                if chemical in added_chemicals:

                    df = pd.read_excel(os.path.join(training_pred_score_path, 'saved_models',
                                       f'{chemical}_False_y_pred_list_df.xlsx'), index_col=0, engine='openpyxl')
                    df.to_csv(os.path.join(training_pred_score_path, 'saved_models',
                              'Predictions', f'predictions_{chemical}.csv'))
                else:
                    df = pd.read_csv(os.path.join(
                        training_pred_score_path, 'saved_models', f'{chemical}_False_y_pred_list_df.csv'), index_col=0)
                    df.to_csv(os.path.join(training_pred_score_path, 'saved_models',
                              'Predictions', f'predictions_{chemical}.csv'))

#             if method == "regression":
            if codes != None:
                df1, df2 = EvaluationTool()._preds_vs_wet_statistics(
                    training_pred_score_path, chemical, codes=codes)

                other_chems = pd.concat([other_chems, df1])
            else:

                df1, df2 = EvaluationTool()._preds_vs_wet_statistics(
                    training_pred_score_path, chemical, codes=None)
                other_chems = pd.concat([other_chems, df1])

        if codes != None:

            path_to_save = os.path.join(
                training_pred_score_path, 'DLv2.2', 'saved_models', 'Evaluation_Summary_subset')
            os.makedirs(path_to_save, exist_ok=True)
            p_df.to_csv(os.path.join(
                path_to_save, f'classes_Evaluation_{region}_subset.csv'))
            other_chems.to_csv(os.path.join(
                path_to_save, f'Evaluation_{region}_subset.csv'))
        else:

            path_to_save = os.path.join(
                training_pred_score_path, 'saved_models', 'Evaluation_Summary')
            os.makedirs(path_to_save, exist_ok=True)

            p_df.to_csv(os.path.join(
                path_to_save, f'classes_Evaluation_{region}.csv'))
            other_chems.to_csv(os.path.join(
                path_to_save, f'Evaluation_{region}.csv'))
#             other_chems =  other_chems.drop_duplicates()

        if codes != None:
            path_to_save = os.path.join(
                training_pred_score_path, 'saved_models')
            df_2 = pd.read_csv(os.path.join(
                path_to_save, 'Evaluation_Summary_subset', f'Evaluation_{region}_subset.csv'), index_col=0)
            print(df_2)
            df_2 = df_2.reset_index().set_index(['index', 'Model'])
            df_2_copy = df_2.copy()
        else:
            path_to_save = os.path.join(
                training_pred_score_path, 'saved_models')
            df_2 = pd.read_csv(os.path.join(
                path_to_save, 'Evaluation_Summary', f'Evaluation_{region}.csv'), index_col=0)
            print(df_2)
            df_2 = df_2.reset_index().set_index(['index', 'Model'])
            df_2_copy = df_2.copy()
        list_of_models = []
        for chemical in chemicals_conv:
            f = EvaluationTool().get_chem_model_map(training_pred_score_path, chemical)
            model = f[chemical]
            list_of_models.extend(model)

        models = list(dict.fromkeys(list_of_models))
        print('+++++++++++MODELSS++++++++++++', models)

#         models = ['DLv2.0','DLv2.2']


#         models =['StackingBayNuSVR','LightGBM', 'BayesianRidge','ExtGradientBoost']
        guides = {
            'boron': [-1e6, 0.5, 0.8, 1, 1e6],
            'calcium_%': [-1e6, 40, 60, 65, 1e6],
            'cec': [-1e6, 8, 15, 20, 1e6],
            'copper': [-1e6, 1, 1.5, 8, 1e6],
            'iron': [-1e6, 20, 30, 50, 1e6],
            'magnesium_%': [-1e6, 8, 10, 15, 1e6],
            'manganese': [-1e6, 10, 20, 100, 1e6],
            'potassium_%': [-1e6, 1.5, 3, 5, 1e6],
            'ph': [-1e6, 5.5, 5.8, 6.4, 1e6],
            'organic_carbon': [-1e6, 1, 2, 4, 1e6],
            'sulphur': [-1e6, 5, 10, 20, 1e6],
            'total_nitrogen': [-1e6, 0.1, 0.2, 0.25, 1e6],
            'zinc': [-1e6, 1, 2, 4, 1e6],
            'phosphorus': [-1e6, 10, 30, 50, 1e6]
        }
#         chemicals = ['organic_carbon','total_nitrogen',  'sand', 'clay', 'silt']
        df_2['PCC0'] = np.nan
        df_2['PCC1'] = np.nan
        df_2['PCC2'] = np.nan
        df_2['PCC3'] = np.nan
        chemicals_conv = list(set(chemicals_conv))

        for chemical in chemicals_conv:
            try:

                for model_first_name in models:
                    print(chemical)

                    df = pd.read_csv(os.path.join(
                        training_pred_score_path, f'{chemical}_{model_first_name}_classes.csv'), index_col=0)

                    df['class_diff'] = abs(
                        df['label_code'] - df['label_preds_code'])

                    total_samples = df.shape[0]

                    value_counts = df.class_diff.value_counts()

                    for class_ in value_counts.index.values:

                        df_2.at[(f'{chemical}', f'{model_first_name}_regression'), f'PCC{class_}'] = round(
                            (value_counts.loc[class_] / total_samples) * 100, 2)
            except:
                pass

# #         if chem_correction:
# #             chemicals_conv = ['zinc','potassium_%','phosphorus']
# #         else:
# #             pass
# #         print(chemicals_conv)
#         df_comb_eval = pd.DataFrame()
#         for chemical in chemicals_conv:


#             for model_first_name in models:
#                 try:


#                     df = pd.read_csv(os.path.join(training_pred_score_path,f'{chemical}_{model_first_name}_classes.csv' ), index_col=0)

#                     df['class_diff'] = abs(df['label_code'] - df['label_preds_code'])

#                     total_samples = df.shape[0]

#                     value_counts = df.class_diff.value_counts()

#                     for class_ in value_counts.index.values:


#                         df_temp = df_2.loc[(f'{chemical}', f'{model_first_name}_regression')]
#                         pcc_stats = round((value_counts.loc[class_] / total_samples) * 100, 2)

#                         df_temp[f'PCC{class_}'] = pcc_stats
#                     df_comb_eval = pd.concat([df_comb_eval,df_temp])
# #                     df_2.at[(f'{chemical}', f'{model_first_name}_regression'), f'PCC{class_}'] = round((value_counts.loc[class_] / total_samples) * 100, 2)
#                 except:
#                    pass
        df_2.index.name = ('index', 'Model')

        reset_chems = ['calcium', 'magnesium', 'potassium']

        df_2.rename(columns={'PCC0': 'Accuracy'}, inplace=True)

        df_2.replace(np.NaN, 0, inplace=True)

#         print(chemical '++++++++++++++++'df_2.columns,'++++++++++++++++')
        df_2 = df_2[['Accuracy', 'PCC1', 'PCC2', 'PCC3']]

        df_2_no_dups = df_2.drop_duplicates(
        ).reset_index().set_index(['index', 'Model'])
        if codes != None:

            #             class_df = pd.read_csv(os.path.join(path_to_save,'Evaluation_Summary_subset', f'classes_Evaluation_{region}_subset.csv'),index_col=0)
            eval_df = pd.read_csv(os.path.join(
                path_to_save, 'Evaluation_Summary_subset', f'Evaluation_{region}_subset.csv'), index_col=0)
        else:
            #             class_df = pd.read_csv(os.path.join(path_to_save,'Evaluation_Summary', f'classes_Evaluation_{region}.csv'),index_col=0)
            class_df = pd.DataFrame()
            eval_df = pd.read_csv(os.path.join(
                path_to_save, 'Evaluation_Summary', f'Evaluation_{region}.csv'), index_col=0)
        eval_df = eval_df.reset_index().set_index(['index', 'Model'])
        if class_df.shape[0] > 0:
            class_df = class_df.reset_index().set_index(['index', 'Model'])
            class_df.drop('no_samples', axis=1, inplace=True)
            df_merged = df_2_no_dups
#             .merge(class_df, how='outer', left_index=True, right_index=True)
            df_merged2 = eval_df.merge(
                df_merged, how='outer', left_index=True, right_index=True)
        else:
            df_merged2 = eval_df.merge(
                df_2_no_dups, how='outer', left_index=True, right_index=True)
        if codes != None:

            df_merged2.to_csv(os.path.join(
                path_to_save, 'Evaluation_Summary_subset', f'Evaluation_{region}_subset.csv'))
        else:
            df_merged2 = df_merged2.drop_duplicates()
            for chem in reset_chems:
                try:
                    df_merged2 = df_merged2.reset_index().set_index('index')

                    df_merged2.at[chem, 'Accuracy'] = np.NaN
                    df_merged2.at[chem, 'PCC1'] = np.NaN
                    df_merged2.at[chem, 'PCC2'] = np.NaN
                    df_merged2.at[chem, 'PCC3'] = np.NaN
                    df_merged2 = df_merged2.reset_index(
                    ).set_index(['index', 'Model'])
                except:
                    pass
            if corrected_chems:
                df_merged2['corrected'] = True
                df_merged2 = df_merged2.reset_index().set_index('index')
                df_merged2 = df_merged2.drop(['calcium', 'magnesium'])
#                 df_merged2 = df_merged2.head(6)
                df_merged2.to_csv(os.path.join(
                    path_to_save, 'Evaluation_Summary', f'Evaluation_{region}.csv'))

            else:
                df_merged2['corrected'] = False
                df_merged2.to_csv(os.path.join(
                    path_to_save, 'Evaluation_Summary', f'Evaluation_{region}.csv'))
            df_guides = pd.DataFrame(guides)
            df_guides = df_guides.head(4)
            df_guides = df_guides.tail(3)
            df_guides.to_csv(os.path.join(
                path_to_save, 'Evaluation_Summary', f'Advice_Guides_{region}.csv'))
        best_models_comb_df = pd.DataFrame()
        chemicals_conv = list(set(chemicals_conv))
        if chem_correction:
            chemicals_conv = ['zinc', 'potassium', 'phosphorus']
        else:
            pass
        for chemical in chemicals_conv:

            #

            if codes != None:
                path_to_evaluation_summary = os.path.join(
                    path_to_save, 'Evaluation_Summary_subset', f'Evaluation_{region}_subset.csv')
                best_model = self.BestModelSelectorPerChem(
                    path_to_evaluation_summary=path_to_evaluation_summary, chemical=chemical, working_metrics=working_metrics)

            else:

                path_to_evaluation_summary = os.path.join(
                    path_to_save, 'Evaluation_Summary', f'Evaluation_{region}.csv')
                if chem_correction:
                    pd.read_csv(path_to_evaluation_summary)
                print(path_to_evaluation_summary)
                if chem_correction:
                    best_model = self.BestModelSelectorPerChem(
                        path_to_evaluation_summary=path_to_evaluation_summary, chemical=chemical, working_metrics=working_metrics, chem_correction=True)
                else:
                    best_model = self.BestModelSelectorPerChem(
                        path_to_evaluation_summary=path_to_evaluation_summary, chemical=chemical, working_metrics=working_metrics, chem_correction=False)
            best_model_df = best_model.replace(best_model.iloc[0][0], chemical)
            best_models_comb_df = pd.concat(
                [best_models_comb_df, best_model_df])

#             except Exception as e:
#                 print(f'{chemical} fails with exception {e}')
#                 pass
#         print(best_models_comb_df)
        best_models_comb_df = best_models_comb_df.rename(columns={0: 'chemical'}).reset_index(
        ).rename(columns={'index': 'Model'}).set_index('chemical')
        path_to_evaluation_summary = os.path.join(
            path_to_save, 'Evaluation_Summary_subset')
        if codes != None:

            best_models_comb_df.to_csv(os.path.join(
                path_to_evaluation_summary, 'best_model.csv'))
            best_models = pd.read_csv(os.path.join(
                path_to_evaluation_summary, 'best_model.csv'), index_col=[0, 1])
            path_to_evaluation_summary = os.path.join(
                path_to_save, 'Evaluation_Summary_subset', f'Evaluation_{region}_subset.csv')
            evaluation_summary_subset = pd.read_csv(
                path_to_evaluation_summary, index_col=[0, 1])

            comb_df = pd.DataFrame()
            for best in range(len(best_models)):
                tup_chem_ml = best_models.iloc[best].name
                evaluation_summary_subset_2 = evaluation_summary_subset.loc[tup_chem_ml].to_frame(
                ).T
                comb_df = pd.concat([comb_df, evaluation_summary_subset_2])
                path_to_evaluation_summary = os.path.join(
                    path_to_save, 'Evaluation_Summary_subset')
                comb_df.to_csv(os.path.join(
                    path_to_evaluation_summary, 'best_model.csv'))
        else:

            path_to_evaluation_summary = os.path.join(
                path_to_save, 'Evaluation_Summary')
            if corrected_chems:
                best_models_comb_df['corrected'] = True
                best_models_comb_df.to_csv(os.path.join(
                    path_to_evaluation_summary, 'best_model.csv'))
            else:
                best_models_comb_df['corrected'] = False
                best_models_comb_df.to_csv(os.path.join(
                    path_to_evaluation_summary, 'best_model.csv'))

            best_models = pd.read_csv(os.path.join(
                path_to_evaluation_summary, 'best_model.csv'), index_col=[0, 1])
            path_to_evaluation_summary = os.path.join(
                path_to_save, 'Evaluation_Summary', f'Evaluation_{region}.csv')

            evaluation_summary = pd.read_csv(
                path_to_evaluation_summary, index_col=[0, 1])

            comb_df = pd.DataFrame()
            for best in range(len(best_models)):
                tup_chem_ml = best_models.iloc[best].name

                evaluation_summary_2 = evaluation_summary.loc[tup_chem_ml].to_frame(
                ).T

                comb_df = pd.concat([comb_df, evaluation_summary_2])
                path_to_evaluation_summary = os.path.join(
                    path_to_save, 'Evaluation_Summary')
#                 comb_df=comb_df.drop_duplicates()

                comb_df.to_csv(os.path.join(
                    path_to_evaluation_summary, 'best_model.csv'))

            folders = ['Evaluation_Summary', 'Predictions',
                       'confusion_matrix', 'scatter_plots']
            path_to_save = os.path.join(
                training_pred_score_path, 'saved_models')

            today = date.today()

            d1 = today.strftime("%Y/%m/%d")
            d1 = d1.replace('/', '_')

            if chem_correction:
                pass
            else:

                os.makedirs(os.path.join(training_pred_score_path,
                            f'{d1}_{project_name}_v2.0_v2.2_v5'), exist_ok=True)
#                 os.makedirs(os.path.join(training_pred_score_path,f'{d1}_{project_name}_v2.0_v2.2_v5'),exist_ok=True)
                final_folder = os.path.join(
                    training_pred_score_path, f'{d1}_{project_name}_v2.0_v2.2_v5', 'pre_correction')
                os.makedirs(final_folder, exist_ok=True)

                for folder in folders:

                    #                 copy_tree(os.path.join(path_to_save, folder), final_folder)

                    current_folder = os.path.join(path_to_save, folder)
                    # !cp -r {current_folder} {final_folder}

    def conform_headers(self, df):
        print('Conforming headers')
        old_heads = list(df)
        new_heads = []
        for feature in old_heads:
            feature = str(feature)
    # #         print('Replacing feature %s with %s', feature, feature.replace(" ", "_").lower())
            new_heads.append(feature.replace(" ", "_"))
    #         print('Replacing index %s with %s', df.index.name, df.index.name.replace(" ", "_").lower())
        df.index.name = df.index.name.replace(" ", "_").lower()
        df.columns = new_heads
        return df, new_heads

    def BestModelSelectorPerChem(self, path_to_evaluation_summary, chemical, working_metrics, All_metrics=False, chem_correction=False):
        eval_summary = pd.read_csv(path_to_evaluation_summary, engine='python')
        eval_summary = eval_summary.loc[eval_summary['Model'].notnull()]
        eval_summary = eval_summary.fillna(0.5)

        eval_summary['slope'] = np.abs(eval_summary['slope'])
        eval_summary['intercept'] = np.abs(eval_summary['intercept'])

        All_metrics_available = ['slope', 'intercept', 'RMSE', 'RSC', 'R2', 'RMSECVQ1',
                                 'RMSECVQ2', 'RMSECVQ3', 'RMSECVQ4', 'Accuracy', 'PCC1', 'PCC2', 'PCC3']
        eval_summary_subset = eval_summary.copy()
        metric_rank_dict = {}
        list_models = []
        print(chemical)
#         print(eval_summary_subset)
        if All_metrics == False:
            working_metrics = working_metrics
        else:
            working_metrics = All_metrics_available
        if chemical in ['clay', 'sand', 'calcium', 'silt', 'potassium', 'magnesium', 'aluminium', 'sodium', 'ec_salts', 'exchangeable_acidity', 'sulphur', 'copper']:
            removed_metrics = ['Accuracy', 'recall_score',
                               'precision_score', 'f1_score', 'PCC1', 'PCC2', 'PCC3']
            for metric in removed_metrics:
                if metric in working_metrics:
                    working_metrics.remove(metric)
                else:
                    pass
        else:
            pass

        print('Subset is', eval_summary_subset)

        for metric in working_metrics:
            try:
                #                 eval_ = pivot_table(eval_summary_subset, values=metric, index=['Unnamed: 0'], columns=['Unnamed: 1'], aggfunc='sum')
                eval_ = pivot_table(eval_summary_subset, values=metric, index=[
                                    'index'], columns=['Model'], aggfunc='sum')
            except:
                #                 eval_ = pivot_table(eval_summary_subset, values=metric, index=['index'], columns=['Model'], aggfunc='sum')

                eval_ = pivot_table(eval_summary_subset, values=metric, index=[
                                    'Unnamed: 0'], columns=['Unnamed: 1'], aggfunc='sum')
#             try:

                eval_, _ = self.conform_headers(eval_)
                eval_ = eval_[(eval_.T != 0).any()]
#             except:
#                 pass

            if metric in ['R2', 'RSC', 'recall_score', 'precision_score', 'f1_score', 'Accuracy', 'slope']:
                #                 print(eval_.T.sort_values(by = chemical, ascending=False)[chemical])
                print(eval_.T)
                list_models = eval_.T.sort_values(by=chemical, ascending=False)[
                    chemical].index.tolist()
            else:
                list_models = eval_.T.sort_values(by=chemical, ascending=True)[
                    chemical].index.tolist()
                # dict with numerical index for different models available
            list_models.extend(list_models)

            # remove duplicates from list
            list_models = list(dict.fromkeys(list_models))
#             print(list_models)
            models_arranged_based_on_performance = dict(
                [(y, x) for x, y in enumerate(list_models)])
#             print(models_arranged_based_on_performance)
            metric_rank_dict.update(
                {(metric): ((models_arranged_based_on_performance))})
#
#
            print(metric_rank_dict)
            df_rank_models = pd.DataFrame(metric_rank_dict)
            print(df_rank_models)

            best_model = pd.DataFrame(df_rank_models.sum(
                axis=1)).sort_values(by=[0]).head(1)


#             if aggregator == 'minimum':

#             else:
#                 pass

        return best_model

    def Different_Models_Eval(self, path_to_file, project_name, training_pred_score_paths=None, wet_chem_path=None, predict=False, codes=None):
        #         for training_pred_score_path in paths_to_models_summary_config:
        df_dict = {}
        list_ = []
        with open(path_to_file, 'r') as fd:
            reader = csv.reader(fd)
            for row in reader:
                list_.append(row)
        for training_pred_score_path in list_:
            try:
                df = pd.read_csv(os.path.join(
                    training_pred_score_path[0], 'saved_models', 'Evaluation_Summary', 'best_model.csv'))
                df['modelling_type'] = training_pred_score_path[1]
                df_dict.update({training_pred_score_path[1]: df})
                self.Models_Summary(training_pred_score_path[0], project_name, working_metrics,
                                    training_pred_score_paths=None, wet_chem_path=None, predict=False, codes=None)
            except:
                pass

        chemicals_conv = []
        training_pred_score_path = list_[0][0]
        path_to_saved_models = os.path.join(
            training_pred_score_path, 'saved_models')
        for i in range(len(os.listdir(path_to_saved_models))):
            if os.listdir(path_to_saved_models)[i].endswith('False_y_pred_list_df.csv'):
                chem_ = os.listdir(path_to_saved_models)[i].split('_')[0]
                chem_, _ = EvaluationTool().correct_chemical_name(chem_)
                chemicals_conv.append(chem_)
            else:
                pass


#             print(training_pred_score_path[0])

        df_5 = pd.concat([df_dict[k] for k, _ in df_dict.items()])

        df_5 = df_5.rename(
            columns={'Unnamed: 0': 'index', 'Unnamed: 1': 'Model'})
        new = df_5["modelling_type"].copy()

        df_5["Model"] = df_5["Model"].str.cat(new, sep=", ")

        All_metrics_available = ['slope', 'intercept', 'RMSE', 'RSC', 'R2', 'RMSECVQ1',
                                 'RMSECVQ2', 'RMSECVQ3', 'RMSECVQ4', 'Accuracy', 'PCC1', 'PCC2', 'PCC3']
#          'recall_score', 'precision_score','f1_score'
        eval_summary_subset = df_5.copy()
        eval_summary_subset = eval_summary_subset.fillna(0.5)
        metric_rank_dict = {}
        list_models = []
        best_models = []
        for chemical in chemicals_conv:
            working_metrics = All_metrics_available
            for metric in working_metrics:

                eval_ = pivot_table(eval_summary_subset, values=metric, index=[
                                    'index'], columns=['Model'], aggfunc='sum')
                eval_, _ = self.conform_headers(eval_)
                eval_ = eval_[(eval_.T != 0).any()]

                if metric in ['R2', 'RSC']:
                    #                     'recall_score','precision_score','f1_score'

                    list_models = eval_.T.sort_values(by=chemical, ascending=False)[
                        chemical].index.tolist()
                else:
                    list_models = eval_.T.sort_values(by=chemical, ascending=True)[
                        chemical].index.tolist()
                    # dict with numerical index for different models available
                list_models.extend(list_models)
                # remove duplicates from list
                list_models = list(dict.fromkeys(list_models))

                models_arranged_based_on_performance = dict(
                    [(y, x) for x, y in enumerate(list_models)])
                metric_rank_dict.update(
                    {metric: models_arranged_based_on_performance})
                df_rank_models = pd.DataFrame(metric_rank_dict)
            #             if aggregator == 'minimum':
                best_model = pd.DataFrame(df_rank_models.sum(
                    axis=1)).sort_values(by=[0]).head(1)
            #

            best_models_comb_df = pd.DataFrame()
            best_model_df = best_model.replace(best_model.iloc[0][0], chemical)
            best_models_comb_df = pd.concat(
                [best_models_comb_df, best_model_df])
            best_models_comb_df = best_models_comb_df.rename(columns={0: 'chemical'}).reset_index(
            ).rename(columns={'index': 'Model'}).set_index('chemical')
            best_models.append(best_models_comb_df)
            # path_to_evaluation_summary= os.path.join(path_to_save,'Evaluation_Summary_subset')

        df_5 = df_5.set_index(['index', 'modelling_type'])
        combined_best_ml = pd.DataFrame()
        for index_no, chemical in enumerate(chemicals_conv):

            df_7 = df_5.loc[(chemical, best_models[index_no].Model.str.split(',')[
                             0][1].strip('_'))].to_frame().T

            combined_best_ml = pd.concat([combined_best_ml, df_7])
#             os.makedirs(f'{os.getcwd()}_{project_name}', exist_ok=True)
#             '/home/java/DS-RL22_Fritsch-grinding-plants/combined_best_model'
            combined_best_ml.to_csv(
                '/home/java/DS-RL22_Fritsch-grinding-plants/combined_best_model/best_models.csv')

    def post_prediction_preds(self, training_pred_score_path, project_name, working_metrics, corrected_chems=None, chem_correction=None):
        region = project_name
        k_correction_formular = "potassium_v1 if < 230 : potassium_prediction * 1.288076 + (-106.6364)"
        zn_correction_formular = "zinc_v1: zinc_prediction - 2"
        phosphorus_correction = 'phosphorus_v3:  [<30, 30, 50, 80, > 80 ]'
#         lines = [k_correction_formular, zn_correction_formular, phosphorus_correction]
        lines = [k_correction_formular, phosphorus_correction]
        source_path = training_pred_score_path
        added_chemicals = ['calcium_%', 'potassium_%', 'magnesium_%']

        chemicals_conv = []
        path = os.path.join(training_pred_score_path, 'post_correction', )
        os.makedirs(path, exist_ok=True)
        path2 = os.path.join(training_pred_score_path,
                             'post_correction', 'saved_models')
        os.makedirs(path2, exist_ok=True)
        path_to_saved_models = os.path.join(
            training_pred_score_path, 'saved_models')
        path_to_saved_models = Path(path_to_saved_models)
        chem_correction = True
        all_models_chems = [x for x in path_to_saved_models.glob(
            '**/*False_y_pred_list_df.csv')]

        for models_chem in all_models_chems:
            chem_ = models_chem.name.split('_False')[0]
            chemicals_conv.append(chem_)

        all_chems = chemicals_conv
        print("These are the chemicals conv$$$$$$$$$$$$$$$$$$", all_chems)
        for chemical in chemicals_conv:

            if chemical in corrected_chems:
                chem_map = EvaluationTool().get_chem_model_map(
                    training_pred_score_path, chemical)
                models = chem_map.get(chemical)
                for idx, models_ in enumerate(chem_map.get(chemical)):
                    print(idx, '+++++++++++++++++++++++++++++++++++++++++')
                    model_first_name = models_

                    print(
                        f'+ + + + + + Starting Post prediction corection for {chemical} + + + + + + +')

                    df = pd.read_csv(os.path.join(
                        training_pred_score_path, 'saved_models/', f'{chemical}_False_y_pred_list_df.csv'), index_col=0)
                    if chemical == 'zinc':

                        if idx == 0:
                            comb_df_corrected = self.zinc_correction(
                                df, models)
                            comb_df_corrected.to_csv(os.path.join(
                                path, 'saved_models/', f'{chemical}_False_y_pred_list_df.csv'))
                            comb_df_corrected.to_pickle(os.path.join(
                                path, 'saved_models/', f'{chemical}_{model_first_name}_regression_False_score_trained.pkl'))
                        else:
                            comb_df_corrected = pd.DataFrame()

                        comb_df_corrected.to_pickle(os.path.join(
                            path, 'saved_models/', f'{chemical}_{model_first_name}_regression_False_score_trained.pkl'))
                    elif chemical == 'potassium':

                        #                         source = os.path.join(source_path ,'saved_models','cec_False_y_pred_list_df.csv')
                        #                         shutil.copy2(source,os.path.join(path,'saved_models/','cec_False_y_pred_list_df.csv'))
                        print(f'running {chemical} post-correction')
                        if idx == 0:

                            comb_df_corrected = self.potassium_correction(
                                df, models)
                            comb_df_corrected.to_csv(os.path.join(
                                path, 'saved_models/', f'{chemical}_False_y_pred_list_df.csv'))
#                             comb_df_corrected.to_pickle(os.path.join(path ,'saved_models/',f'cec_{model_first_name}_regression_False_score_trained.pkl'))
#                             EvaluationTool().chem_to_percentage_conversion(path,corrected_chems)

                        else:
                            comb_df_corrected = pd.DataFrame()

#
                        comb_df_corrected.to_pickle(os.path.join(
                            path, 'saved_models/', f'{chemical}_{model_first_name}_regression_False_score_trained.pkl'))
#                         comb_df_corrected.to_pickle(os.path.join(path ,'saved_models/',f'cec_{model_first_name}_regression_False_score_trained.pkl'))


#                                 else:
#                                     pass
                    elif chemical == 'phosphorus':
                        source = os.path.join(
                            source_path, 'saved_models', f'{chemical}_False_y_pred_list_df.csv')
                        shutil.copy2(source, os.path.join(
                            path, 'saved_models/', f'{chemical}_False_y_pred_list_df.csv'))
                        df.to_pickle(os.path.join(
                            path, 'saved_models/', f'{chemical}_{model_first_name}_regression_False_score_trained.pkl'))

            else:
                pass

        with open(os.path.join(path2, 'post_prediction_rules.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

        self.Models_Summary(path, project_name, working_metrics,
                            chem_correction, corrected_chems)

        best_models_comb_df = pd.DataFrame()
#         chemicals_conv.extend(added_chemicals)
        path_to_evaluation_non_corr = os.path.join(
            training_pred_score_path, 'saved_models', 'Evaluation_Summary', f'best_model.csv')
        path_to_evaluation_corr = os.path.join(
            path2, 'Evaluation_Summary', f'best_model.csv')
        df_corr = pd.read_csv(path_to_evaluation_corr)
        df_corr = df_corr.rename(
            columns={'Unnamed: 1': 'Model', 'Unnamed: 0': 'index'})
        df_corr['Model'] = [df_corr['Model'][x] +
                            '_corrected' for x in range(df_corr.shape[0])]
        df_corr = df_corr.set_index(['index', 'Model'])
        df_non_corr = pd.read_csv(
            path_to_evaluation_non_corr, index_col=[0, 1])
        df_comb_corr_non = pd.concat([df_corr, df_non_corr])
        df_comb_corr_non.to_csv(os.path.join(path2, 'best_model.csv'))
        df_comb_corr_non.to_csv(os.path.join(
            path2, f'Evaluation_{region}.csv'))

        path_to_evaluation_summary = os.path.join(path2, 'best_model.csv')
#         for chem in all_chems:
#             if 'calcium' == chem:
#                 chem = 'calcium_%'
#                 all_chems.append(chem)
#             elif 'potassium' == chem:
#                 chem = 'potassium_%'
#                 all_chems.append(chem)
#             elif 'magnesium' == chem:
#                 chem = 'magnesium_%'
#                 all_chems.append(chem)

        for chemical in all_chems:
            try:
                print('Printing is +++++++++++++++++++++++++++++++++++++++++', chemical)

                best_model = self.BestModelSelectorPerChem(
                    path_to_evaluation_summary=path_to_evaluation_summary, chemical=chemical, working_metrics=working_metrics)
                best_model_df = best_model.replace(
                    best_model.iloc[0][0], chemical)
                best_models_comb_df = pd.concat(
                    [best_models_comb_df, best_model_df])

            except:
                pass
                #             except Exception as e:
            #                 print(f'{chemical} fails with exception {e}')
            #                 pass
    #         print(best_models_comb_df)

        best_models_comb_df = best_models_comb_df.rename(columns={0: 'chemical'}).reset_index(
        ).rename(columns={'index': 'Model'}).set_index('chemical')


#         path_to_evaluation_summary= os.path.join(path_to_save)
        best_models_comb_df.to_csv(os.path.join(path2, 'best_model_2.csv'))

        best_models = pd.read_csv(os.path.join(
            path2, 'best_model_2.csv'), index_col=[0, 1])
        path_to_evaluation_summary = os.path.join(
            path2, f'Evaluation_{region}.csv')
        evaluation_summary = pd.read_csv(
            path_to_evaluation_summary, index_col=[0, 1])

        comb_df = pd.DataFrame()
        os.makedirs(os.path.join(path_to_saved_models,
                    'best_models'), exist_ok=True)
        path_to_save = os.path.join(path_to_saved_models, 'best_models')
        for best in range(len(best_models)):
            tup_chem_ml = best_models.iloc[best].name
#             try:
            evaluation_summary_2 = evaluation_summary.loc[tup_chem_ml].to_frame(
            ).T
#                 return evaluation_summary_2
#             except:
#                 evaluation_summary_2 = evaluation_summary.loc[tup_chem_ml]
#                 return evaluation_summary_2
#             evaluation_summary_2 = evaluation_summary_2.reset_index()
            comb_df = pd.concat([comb_df, evaluation_summary_2])
            path_to_evaluation_summary = os.path.join(path_to_save)
        #                 comb_df=comb_df.drop_duplicates()
        #     if corrected_chems:
        #         comb_df['corrected'] = True
        #     else:

        #         comb_df['corrected'] = False
            comb_df = comb_df.drop_duplicates()
            comb_df.to_csv(os.path.join(
                path_to_evaluation_summary, 'best_model.csv'))

    def zinc_correction(self, df, models):
        df_y_true_val = df['y_true_val'].to_frame()
        comb_df_corrected = pd.DataFrame()

    #     print(f'running {chemical} post-correction')
        for model_first_name in models:
            for index in df.index:
                print(model_first_name)
                zn_pred = df.at[index, f'{model_first_name}_regression']
                print(f'subtracting 2...')
                zn_pred_corrected = zn_pred - 2
                print(f'replacing {zn_pred} with {zn_pred_corrected}')
                df.at[index,
                      f'{model_first_name}_regression'] = zn_pred_corrected

                print('done')
            df_ = df[f'{model_first_name}_regression'].to_frame()
            comb_df_corrected = pd.concat([comb_df_corrected, df_], axis=1)
        comb_df_corrected = pd.concat(
            [comb_df_corrected, df_y_true_val], axis=1)
        return comb_df_corrected

    def potassium_correction(self, df, models):
        models = list(dict.fromkeys(models))
        df_y_true_val = df['y_true_val'].to_frame()
        comb_df_corrected = pd.DataFrame()
        chemical = 'potassium'
    #     print(f'running {chemical} post-correction')
        for model_first_name in models:
            for index in df.index:
                k_pred = df.at[index, f'{model_first_name}_regression']
                if k_pred < 230:
                    print(f'{chemical} prediction less than 230, correcting...')
                    k_pred_corrected = k_pred * 1.288076+(-106.6364)
                    print(f'replacing {k_pred} with {k_pred_corrected}')
                    df.at[index,
                          f'{model_first_name}_regression'] = k_pred_corrected
                    print('done')
            df_ = df[f'{model_first_name}_regression'].to_frame()
            comb_df_corrected = pd.concat([comb_df_corrected, df_], axis=1)
        comb_df_corrected = pd.concat(
            [comb_df_corrected, df_y_true_val], axis=1)
        return comb_df_corrected

    def ModelsSummaryStats(self, training_pred_score_path, project_name, working_metrics, corrected_chems=None, chem_correction=None, training_pred_score_paths=None, wet_chem_path=None, predict=False, codes=None):

        self.Models_Summary(training_pred_score_path, project_name,
                            working_metrics, corrected_chems=False)

        df = self.post_prediction_preds(
            training_pred_score_path, project_name, working_metrics, corrected_chems, chem_correction=True)
        folders = ['Evaluation_Summary', 'Predictions',
                   'confusion_matrix', 'scatter_plots']

        today = date.today()

        d1 = today.strftime("%Y/%m/%d")
        d1 = d1.replace('/', '_')
#         os.makedirs(os.path.join(training_pred_score_path,f'{d1}_{project_name}_v2.0_v2.2_v5'),exist_ok=True)
#                 os.makedirs(os.path.join(training_pred_score_path,f'{d1}_{project_name}_v2.0_v2.2_v5'),exist_ok=True)
        final_folder = os.path.join(
            training_pred_score_path, f'{d1}_{project_name}_v2.0_v2.2_v5', 'post_correction')
        os.makedirs(final_folder, exist_ok=True)

        path_to_save = os.path.join(
            training_pred_score_path, 'post_correction', 'saved_models')
        post_prediction_rules_path = os.path.join(
            training_pred_score_path, 'post_correction', 'saved_models', 'post_prediction_rules.txt')

        for folder in folders:

            #                 copy_tree(os.path.join(path_to_save, folder), final_folder)

            current_folder = os.path.join(path_to_save, folder)
            # !cp -r {current_folder} {final_folder}
        best_models_folder = os.path.join(
            training_pred_score_path, 'saved_models', 'best_models')
        final_folder_models = os.path.join(
            training_pred_score_path, f'{d1}_{project_name}_v2.0_v2.2_v5')
        # !cp -r {best_models_folder } {final_folder_models}
        shutil.copy2(post_prediction_rules_path, final_folder)
        return df


def eval(chemicals, path_to_spectra, path_to_wet, predction_folder_path, model_versions, path_to_model, output_path, account_username):
    # v2.0 v2.2
    import gc
    gc.collect()

    global Models_Summary
    Models_Summary = Models_Summary()
    # chemicals = ['magnesium','zinc', 'manganese', 'sodium', 'potassium', 'sulphur', 'calcium','ph', 'phosphorus', 'clay', 'sand', 'silt', 'aluminium', 'boron', 'copper','iron']
    # chemicals = ['exchangeable_acidity']

    # path_to_spectra = 'D://CropNutsDocuments/DS-ML87/outputFiles/data/spc/spc.csv'

    # path_to_wet = 'D://CropNutsDocuments/DS-ML87/outputFiles/data/wetchem/wetchem.csv'
    # predction_folder_path = Path('D://CropNutsDocuments/DS-ML87/outputFiles/data/preds')
    # model_versions = ['DLv2.0','DLv2.2']
    # model_versions = ['DLv2.3']

    # path_to_model = 'D://CropNutsDocuments/DS-ML87/outputFiles/exchangeable_acidity_20230502_090639.071097'
    # output_path = 'D://CropNutsDocuments/DS-ML87/outputFiles/data/preds'
    predict_chems(path_to_model, predction_folder_path, chemicals,
                  model_versions, pd.read_csv(path_to_spectra, engine='c', index_col=0))
    post_pred_version_per_chem = {
        'potassium': 'v1', 'phosphorus': 'v3', 'zinc': 'v1'}
    paths = [path_to_spectra]
    # join_diff_models_data(output_path ,model_versions, chemicals, "OCP_NG")
    join_preds_wet(path_to_wet, output_path, model_versions,
                   chemicals, predction_folder_path)
    working_metrics = ['slope', 'intercept', 'RMSE', 'RSC', 'R2', 'RMSECVQ1',
                       'RMSECVQ2', 'RMSECVQ3', 'RMSECVQ4', 'Accuracy', 'PCC1', 'PCC2', 'PCC3']
    # 'recall_score', 'precision_score','f1_score'
    from pandas import pivot_table
    training_pred_score_path = f'/home/{account_username}/DSML125/outputFiles/predictions'
    lines = [chemicals, path_to_spectra, path_to_wet, predction_folder_path, model_versions,
             path_to_model, output_path, post_pred_version_per_chem, paths, working_metrics]
    delete_files(training_pred_score_path)
    # project_name = 'v2.0-v2.2'
    df = Models_Summary.ModelsSummaryStats(training_pred_score_path, 'ModelUpdateTrial', working_metrics, corrected_chems=[
        'zinc', 'phosphrous'], chem_correction=True, predict=False)

    # 2021-x-x_v2.0-v2.2_v2.0-v2.2_v5.1

    project_name = 'v2.2'
    lines_dict = {0: 'chemicals', 1: 'path_to_spectra', 2: 'path_to_wet', 3: "prediction_folder_path", 4: "model_versions",
                  5: "path_to_model", 6: "output_path", 7: "post_pred_version_per_chem", 8: "paths", 9: 'working_metrics'}
    delete_uneccessay_files(training_pred_score_path)
    saved_eval_params(training_pred_score_path,
                      project_name, lines, lines_dict)

    import gc
    gc.collect()
