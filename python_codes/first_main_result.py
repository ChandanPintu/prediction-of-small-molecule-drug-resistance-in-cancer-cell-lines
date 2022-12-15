import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from scipy import stats


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_rmse_rp_values_one_cell_line(cell, hx_gx, dir_cell_in, confidence_lvl=None, most_potent=None):
    data_in = dir_cell_in + hx_gx + '/'
    if confidence_lvl is not None:
        data_in = data_in + 'confidence_' + confidence_lvl + \
            '/cell_line_' + cell + '_CL_' + confidence_lvl + '.feather'
        df_one_cell = pd.read_feather(data_in)
        df_one_cell['correct'] = np.where(
            (df_one_cell['Observed pGI50'] >= df_one_cell['min']) & (
                df_one_cell['Observed pGI50'] <= df_one_cell['max']), 'with CP', 'without CP')
        df_one_cell = df_one_cell.loc[df_one_cell.correct == 'with CP']
    else:
        data_in = data_in + 'confidence_80/cell_line_' + cell + '_CL_80.feather'
        df_one_cell = pd.read_feather(data_in)
    if most_potent is not None:
        df_one_cell = df_one_cell[df_one_cell['Observed pGI50'] >= most_potent]
    # prediction value
    prediction = df_one_cell.loc[:, 'Predicted pGI50'].to_numpy()
    y = df_one_cell.loc[:, 'Observed pGI50'].to_numpy()  # true value
    rmse_in_cell = rmse(y, prediction)
    rp_in_cell, omit = stats.pearsonr(y, prediction)
    n_rows, n_cols = df_one_cell.shape
    return rmse_in_cell, rp_in_cell, n_rows


def plot_median_rmse_rp(df, y, most_potent):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = 13
    sns.scatterplot(data=df,
                    x='Number of test set molecules', y=y,
                    hue='Underlying model', style="Type of validation", s=85,
                    markers=["X", "^", "d", "s", "H"], ax=ax)
    plt.grid(which='both', linestyle=':', linewidth=0.6)
    plt.legend()
    plt.minorticks_on()
    cwd = os.getcwd()
    dir_out_images = cwd + '/images/median/'
    if not (os.path.isdir(dir_out_images)):
        os.makedirs(dir_out_images, exist_ok=True)
    if most_potent is not None:
        name_save_fig = dir_out_images + 'median_' + y + '_most_potent.png'
    else:
        name_save_fig = dir_out_images + 'median_' + y + '.png'
    plt.savefig(name_save_fig, bbox_inches='tight')
    return None


def read_60_test_sets(hx_gx, dir_cell_in, most_potent=None):
    cells = ["NCI-H23", "NCI-H522", "A549_ATCC", "EKVX", "NCI-H226", "NCI-H322M", "NCI-H460", "HOP-62", "HOP-92",
             "HT29", "HCC-2998", "HCT-116", "SW-620", "COLO_205", "HCT-15", "KM12", "MCF7", "NCI_ADR-RES",
             "MDA-MB-231_ATCC", "HS_578T", "MDA-MB-435", "BT-549", "T-47D", "OVCAR-3", "OVCAR-4", "OVCAR-5",
             "OVCAR-8", "IGROV1", "SK-OV-3", "CCRF-CEM", "K-562", "MOLT-4", "HL-60(TB)", "RPMI-8226", "SR",
             "UO-31", "SN12C", "A498", "CAKI-1", "RXF_393", "786-0", "ACHN", "TK-10", "LOX_IMVI", "MALME-3M",
             "SK-MEL-2", "SK-MEL-5", "SK-MEL-28", "M14", "UACC-62", "UACC-257", "PC-3", "DU-145", "SNB-19",
             "SNB-75", "U251", "SF-268", "SF-295", "SF-539", "MDA-N"]
    rmse_hx_gx, rp_hx_gx, num_mol_hx_gx = [], [], []
    for confidence in [None, '80', '85', '90', '95']:
        rmse_cl, rp_cl, num_cl = [], [], []
        for cell in cells:
            rmse_cell, rp_cell, n_mol_cell = get_rmse_rp_values_one_cell_line(cell=cell, hx_gx=hx_gx,
                                                                              dir_cell_in=dir_cell_in,
                                                                              confidence_lvl=confidence,
                                                                              most_potent=most_potent)
            rmse_cl.append(rmse_cell)
            rp_cl.append(rp_cell)
            num_cl.append(n_mol_cell)
        rmse_hx_gx.append(rmse_cl)
        rp_hx_gx.append(rp_cl)
        num_mol_hx_gx.append(num_cl)
    return rmse_hx_gx, rp_hx_gx, num_mol_hx_gx


def median_rmse_rp_cp_models(dir_cell_in, most_potent=None):
    print('\nCP model: RF-RF \n\tReading predictions... ')
    rmse_RF, rp_RF, num_mol_RF = read_60_test_sets(
        hx_gx='RF-RF', dir_cell_in=dir_cell_in, most_potent=most_potent)
    df_med_RF = pd.DataFrame(
        data=['non-CP', '80%', '85%', '90%', '95%'], columns=['Type of validation'])
    df_med_RF['Number of test set molecules'] = np.median(
        np.array(num_mol_RF), axis=1)
    df_med_RF['Underlying model'] = 'h(x): RF'
    df_med_RF['RMSE'] = np.median(np.array(rmse_RF), axis=1)
    df_med_RF['Rp'] = np.median(np.array(rp_RF), axis=1)
    print('CP model: XGB-RF \n\tReading predictions... ')
    rmse_XGB, rp_XGB, num_mol_XGB = read_60_test_sets(
        hx_gx='XGB-RF', dir_cell_in=dir_cell_in, most_potent=most_potent)
    df_med_XGB = pd.DataFrame(
        data=['non-CP', '80%', '85%', '90%', '95%'], columns=['Type of validation'])
    df_med_XGB['Number of test set molecules'] = np.median(
        np.array(num_mol_XGB), axis=1)
    df_med_XGB['Underlying model'] = 'h(x): XGB'
    df_med_XGB['RMSE'] = np.median(np.array(rmse_XGB), axis=1)
    df_med_XGB['Rp'] = np.median(np.array(rp_XGB), axis=1)
    df_med_both = pd.concat([df_med_RF, df_med_XGB], ignore_index=True)
    print('Plot: median RMSE ...')
    plot_median_rmse_rp(df=df_med_both, y='RMSE', most_potent=most_potent)
    print('Plot: median Rp ...')
    plot_median_rmse_rp(df=df_med_both, y='Rp', most_potent=most_potent)
    return None
