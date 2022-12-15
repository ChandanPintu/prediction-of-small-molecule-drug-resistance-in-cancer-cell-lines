from first_main_result import median_rmse_rp_cp_models
from second_main_result import make_predictions, plot_observed_vs_predicted


def run_first_main_result(dir_data_in):
    print('--- ' * 25, '\nRunning: Median RMSE and Rp  values in the 60 test sets (one per cell line)\n',
          '--- ' * 25)
    dir_data_in = dir_data_in + 'CP_NCI60_data/test_predictions/'
    median_rmse_rp_cp_models(dir_cell_in=dir_data_in, most_potent=None)
    median_rmse_rp_cp_models(dir_cell_in=dir_data_in, most_potent=6)
    return None


def run_second_main_result(cell_line, hx, gx, dir_data_in, confidence_level=80):
    print('--- ' * 25, '\nRunning: Observed and predicted pGI50 value in the best/worst predicted cell line\n',
          '--- ' * 25, '\nCell line: ', cell_line)
    df_prediction = make_predictions(cell=cell_line, hx=hx, gx=gx, confidence=confidence_level,
                                     dir_data_in=dir_data_in)
    print('Plot figure...\n')
    plot_observed_vs_predicted(cell=cell_line, df_cell_line=df_prediction, title_plot=hx+'-'+gx,
                               save_fig=True)
    return None


dir_working = 'D:/MS_Classes/Third_Term_Class/Biomedical/CP_NCI60_project/'
# dir_working= r'D:\MS_Classes\Third_Term_Class\Biomedical\CP_NCI60_project\\'


run_first_main_result(dir_data_in=dir_working)


# run_second_main_result(cell_line="OVCAR-5", hx='RF',
#                        gx='RF', dir_data_in=dir_working, confidence_level=80)
# run_second_main_result(cell_line="SR", hx='RF', gx='RF',
#                        dir_data_in=dir_working, confidence_level=80)
# run_second_main_result(cell_line="NCI-H322M", hx='XGB',
#                        gx='RF', dir_data_in=dir_working, confidence_level=80)
# run_second_main_result(cell_line="SR", hx='XGB', gx='RF',
#                        dir_data_in=dir_working, confidence_level=80)
