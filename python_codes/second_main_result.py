import numpy as np
import pandas as pd
import dill
import os
import matplotlib.pyplot as plt
import seaborn as sns


def make_predictions(cell, hx, gx, confidence, dir_data_in):
    cwd = os.getcwd()
    model_name_save = dir_data_in + '/CPmodels/' + 'model_' + hx + '-' + gx + '_' + cell + '.pkl'
    model_open = open(model_name_save, 'rb')
    icp_n = dill.load(model_open)
    model_open.close()
    print('Model loaded: ', hx, '-', gx)
    # Load test data and activity label
    print('\t Load test data and activity label...')
    dir_data_test = dir_data_in + 'CP_NCI60_data/test_best_worst/'
    x_test = np.loadtxt(dir_data_test + cell + '.test.csv', delimiter=',')
    y_test = np.loadtxt(dir_data_test + cell + '.test.act')
    # PREDICTION
    s = round(1 - confidence / 100, 2)
    prediction_interval, y_predicted, error_predicted, index_score, nc_table = icp_n.predict(x_test, significance=s)
    # Building output
    header = ['min', 'max', 'size', 'Observed pGI50', 'Predicted pGI50', 'Predicted error']
    size_n = prediction_interval[:, 1] - prediction_interval[:, 0]
    table_n = np.vstack([prediction_interval.T, size_n.T, y_test, y_predicted, error_predicted]).T
    df_n = pd.DataFrame(table_n, columns=header)
    # Add new columns to dataframe: Error, non-conformity score (alpha_s) and error_range
    df_n['Observed error'] = pd.Series.abs(df_n['Observed pGI50'] - df_n['Predicted pGI50'])
    df_n['nc_score'] = index_score  # Index score is a single number (repeated for each test instance)
    df_n['error_range'] = df_n['Predicted error'] * df_n['nc_score']
    new_order = ['Observed pGI50', 'Predicted pGI50', 'Predicted error', 'Observed error',
                 'nc_score', 'error_range', 'min', 'max', 'size']
    df_pred = df_n.loc[:, new_order]
    # # # Add CP-valid information
    df_pred['CP-valid prediction'] = np.where((df_pred['Observed pGI50'] >= df_pred['min']) & (df_pred['Observed pGI50'] <= df_pred['max']),
                                              'with CP', 'without CP')
    # # # Add Measured pGI50 information (for the error plot)
    df_pred['Measured pGI50'] = np.where((df_pred["Observed pGI50"] >= 6), 'pGI50 >= 6', 'pGI50 < 6')
    return df_pred


def plot_observed_vs_predicted(cell, df_cell_line, title_plot, save_fig=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams['font.size'] = 14
    sns.scatterplot(data=df_cell_line, x="Observed pGI50", y="Predicted pGI50", hue="CP-valid prediction",
                    hue_order=['with CP', 'without CP'], ax=ax, alpha=0.5, )
    plt.title(title_plot + ': ' + cell)
    plt.axline((4, 4), (6, 6), linewidth=0.7, color='k', linestyle='--')  # identity line
    plt.axvline(x=6, color="black", linestyle="--", linewidth=0.9)  # most potent molecules threshold
    plt.axhline(y=6, color="black", linestyle="--", linewidth=0.9)
    plt.xlim(3.8, 11.2)
    plt.ylim(3.8, 11.2)
    plt.legend(loc='upper left')
    if save_fig:
        cwd = os.getcwd()
        dir_out_images = cwd + '/images/scatter/'
        if not (os.path.isdir(dir_out_images)):
            os.makedirs(dir_out_images, exist_ok=True)
        name_save_fig = dir_out_images + title_plot + '_' + cell + '_obsVSpred.png'
        plt.savefig(name_save_fig, bbox_inches='tight')
    return None

