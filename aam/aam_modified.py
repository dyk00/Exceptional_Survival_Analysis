""" Advanced Alert Monitor (AAM) model

Reproduction of the AAM model orgininally contructed by P. Kipnis in 'Development and validation of an electronic medical  record-based alert score for detection of inpatient deterioration outside the ICU'.

Author: Tom Bakkes

To run the EMM algorithm, "<variables_name>_1" were introduced, ensuring that no missing values were present.
Edited by Dayeong Kim
"""
import os 
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pdb # Only used for debugging

## Define Functions

def get_admit(dataset):
    admit1_b = dataset['ADMIT1'] * 0.31685
    admit2_b = dataset['ADMIT2'] * -0.26948
    admit3_b = dataset['ADMIT3'] * -0.06884
    admit4_b = dataset['ADMIT4'] * 0.021467

    return(pd.Series(admit1_b, name = 'ADMIT1_B'), pd.Series(admit2_b, name = 'ADMIT2_B'), 
           pd.Series(admit3_b, name = 'ADMIT3_B'), pd.Series(admit4_b, name = 'ADMIT4_B'))

def get_power1(var, beta, miss_val, avg_val, std_val, varname):
    var_np = var.values
    var_np[np.isnan(var_np)] = miss_val
    var_imputed = pd.Series(var_np, index=var.index, name=f"{varname}_1")
    var_s = (var_np - avg_val) / std_val
    var_b = var_s * beta
    return pd.Series(var_b, name = f'{varname}_B'), var_s, var_imputed

def get_power2(var, beta, beta2, miss_val, avg_val, avg_val2, std_val, std_val2, varname):
    VAR_B, var_s, var_imputed = get_power1(var, beta, miss_val, avg_val, std_val, varname)
    var2_t = var_s**2
    var2_s = (var2_t - avg_val2) / std_val2
    var2_b = var2_s * beta2
    return VAR_B, pd.Series(var2_b, name = f'{varname}2_B'), var_s, var_imputed

def get_power3(var, beta, beta2, beta3, miss_val, avg_val, avg_val2, avg_val3, 
               std_val, std_val2, std_val3, varname):
    VAR_B, VAR2_B, var_s, var_imputed = get_power2(var, beta, beta2, miss_val, avg_val, avg_val2, std_val, std_val2, varname)
    var3_t = var_s**3
    var3_s = (var3_t - avg_val3) / std_val3
    var3_b = var3_s * beta3
    return VAR_B, VAR2_B, pd.Series(var3_b, name = f'{varname}3_B'), var_imputed

def get_careorder(dataset):
    care_orderdnr_b = dataset['CARE_ORDERDNR'] * -0.73415
    care_orderfull_code_b = dataset['CARE_ORDERFULL_CODE'] * 0.60913
    care_orderpartial_b = dataset['CARE_ORDERPARTIAL'] * 0.12502
    return(pd.Series(care_orderdnr_b, name = 'CARE_ORDERDNR_B'), 
           pd.Series(care_orderfull_code_b, name = 'CARE_ORDERFULL_CODE_B'), 
           pd.Series(care_orderpartial_b, name = 'CARE_ORDERPARTIAL_B'))

def get_daytime(dataset):
    daytime1_b = dataset['DAYTIME1'] * 0.031665
    daytime2_b = dataset['DAYTIME2'] * 0.17697
    daytime3_b = dataset['DAYTIME3'] * -0.00134
    daytime4_b = dataset['DAYTIME4'] * -0.2073
    return(pd.Series(daytime1_b, name = 'DAYTIME1_B'),
           pd.Series(daytime2_b, name = 'DAYTIME2_B'),
           pd.Series(daytime3_b, name = 'DAYTIME3_B'),
           pd.Series(daytime4_b, name = 'DAYTIME4_B'))

def get_miss_trop(dataset):
    miss_trop0_B = np.invert(dataset['MISS_TROP']) * 0.078542
    miss_trop1_B = dataset['MISS_TROP'] * -0.07854
    return(pd.Series(miss_trop0_B, name = 'MISS_TROP0_B'),
           pd.Series(miss_trop1_B, name = 'MISS_TROP1_B'))

def get_neuro(dataset):
    neuro_b = np.zeros(dataset.shape[0])
    neuro_b[dataset['AVPU']==0] = -0.14903
    neuro_b[dataset['AVPU']==1] = 0.11695
    neuro_b[dataset['AVPU']==2] = 0.11695
    neuro_b[dataset['AVPU']==3] = 0.03208
    neuro_b[np.isnan(dataset['AVPU'])] = -0.14903
    return(pd.Series(neuro_b, name = 'NEURO_B'))

def get_season(dataset):
    season_b = np.zeros(dataset.shape[0])
    season_b[dataset['SEASON1']] = 0.013593
    season_b[dataset['SEASON2']] = -0.0224
    season_b[dataset['SEASON3']] = 0.008802
    return(pd.Series(season_b, name = 'SEASON_B'))

def get_sex(dataset):
    sex_b = np.zeros(dataset.shape[0])
    sex_b[dataset['SEX'].values] = 0.058334
    sex_b[np.invert(dataset['SEX'].values)] = -0.05833
    return(pd.Series(sex_b, name = 'SEX_B'))

def get_AAM(dataset):
    ADMIT1_B, ADMIT2_B, ADMIT3_B, ADMIT4_B = get_admit(dataset)

    ANIONGAP_B, _, ANIONGAP_1 = get_power1(dataset['ANIONGAP'].copy()
                               , 0.011762, 8, 8.0859142, 3.193268, 'ANIONGAP')
    BICARB_B, BICARB2_B, _, BICARB_1 = get_power2(dataset['BICARB'].copy()
                                        , -0.027615, 0.077649, 27, 26.7063492, 1, 3.9165285, 2.2207588, 'BICARB')
    BPDIA_B, BPDIA2_B, _, BPDIA_1 = get_power2(dataset['BPDIA'].copy()
                                      , -0.042413, 0.067076, 70, 68.9089773, 1, 12.184098, 1.5237996, 'BPDIA')
    BPSYS_I_B, _, BPSYS_I_1 = get_power1(dataset['BPSYS_I']
                           , 0.10469, 0, 31.0243532, 18.159921, 'BPSYS_I')
    BPSYS_B, BPSYS2_B, BPSYS3_B, BPSYS_1 = get_power3(dataset['BPSYS'].copy(),
                                             -0.06628, 0.21469, -0.091944, 110, 126.0288245, 1, 0.4175758, 
                                             19.5940435, 1.4239064, 4.5074954, 'BPSYS')
    CARE_ORDERDNR_B, CARE_ORDERFULL_CODE_B, CARE_ORDERPARTIAL_B = get_careorder(dataset)

    DAYTIME1_B, DAYTIME2_B, DAYTIME3_B, DAYTIME4_B = get_daytime(dataset)

    GLUCOSE_B, _, GLUCOSE_1 = get_power1(dataset['GLUCOSE'].copy() * 18.02
                           , 0.060638, 100, 116.089368, 41.2232443, 'GLUCOSE')

    HEMAT_B, HEMAT2_B, HEMAT3_B, HEMAT_1 = get_power3(dataset['HEMAT'].copy() * 100
                                             , 0.028566, 0.06859, -0.051343, 34 
                                             , 33.5408798, 1, 0.2330322 
                                             , 5.2495619, 1.53005, 4.6207032, 'HEMAT')

    HRTRT_B, HRTRT2_B, HRTRT3_B, HRTRT_1 = get_power3(dataset['HRTRT'].copy()
                                             , 0.26843, 0.15028, -0.063602, 80
                                             , 79.5397633, 1, 0.5033025
                                             , 14.9681737, 1.6261662, 6.1132073, 'HRTRT')

    LACT_B, _, LACT_1 = get_power1(dataset['LACT'].copy(), 
                           0.068798, 0, 0.3045566, 0.6360381, 'LACT')

    LAPS2_B, LAPS22_B, LAPS23_B, LAPS2_1 = get_power3(dataset['LAPS2'].copy()
                                             , 0.54853, -0.081226, -0.064844, 0
                                             , 62.2147154, 1, 0.674725
                                             , 32.9662004, 1.4985656, 4.6646392, 'LAPS2')

    LAPS2_HET_B, _, LAPS2_HET_1 = get_power1(dataset['LAPS2_HET'].copy(), 
                             -0.19021, 0, 61.9812093, 40.0106276, 'LAPS2_HET')

    LBUN_B, _, LBUN_1 = get_power1(np.log((dataset['BUN'].copy() * 2.8) + 1)
                           , 0.1596, np.log(10 + 1), 2.8353456, 0.6306933, 'LBUN')

    LCREAT_B, LCREAT2_B, _, LCREAT_1 = get_power2(np.log((dataset['CREAT'].copy() / 88.42) + 0.1)
                                        , -0.056173, -0.016519, np.log(1.3 + 0.1), 0.1275647
                                        , 1, 0.5157403, 2.3446488, 'LCREAT')

    LELOS_B, _, LELOS_1 = get_power1(np.log(dataset['ELOS'].copy())
                          , -0.25548, 0, 3.8426625, 1.1247647, 'LELOS')

    ELOSLAPS2_B, _, ELOSLAPS2_1 = get_power1(np.log(dataset['ELOS'].copy()) * dataset['LAPS2'].copy()
                                , 0.001316962, 0, 240.1371679, 151.47896, 'LELOSLAPS2')

    LHRTRT_I_B, LHRTRT2_I_B, _, LHRTRT_I_1 = get_power2(np.log(dataset['HRTRT_I'].copy() + 1)
                                     , 0.018804, 0.16519, 0
                                     , 2.8171576, 1, 0.7999374, 2.3117584, 'LHRTRT_I')

    LSAT_I_B, _, LSAT_I_1 = get_power1(np.log(dataset['SAT_I'].copy() + 1)
                            , 0.11247, 0, 1.502558, 0.593402, 'LSAT_I') 

    NORM_AGE = (dataset['AGE'].copy() - 17) / 100
    NORM_AGE[NORM_AGE <= 0] = 0.01
    NORM_AGE[NORM_AGE >= 1] = 0.99
    LOGIT_AGE_B, LOGIT_AGE2_B, _, LOGIT_AGE_1 = get_power2(np.log(NORM_AGE / (1 - NORM_AGE))
                                             , -0.16634, -0.16443, 0.08
                                             , 0.0834582, 1, 0.8629652, 2.4255876, 'LOGIT_AGE')

    NORM_SAT = (dataset['SAT'].copy() - 1) / 100
    NORM_SAT[NORM_SAT <= 0] = 0.01
    NORM_SAT[NORM_SAT >= 1] = 0.99
    LOGIT_SAT_B, LOGIT_SAT2_B, LOGIT_SAT3_B, LOGIT_SAT_1 = get_power3(np.log(NORM_SAT / (1 - NORM_SAT))
                                                        , -0.03599, 0.1474, 0.046093, np.log(0.99 / (1 - 0.99))
                                                        , 3.2926495, 1, 0.622524
                                                        , 0.6700289, 1.2821673, 3.0441072, 'LOGIT_SAT')

    NORM_SAT_W = (dataset['SAT_W'].copy() - 1) / 100
    NORM_SAT_W[NORM_SAT_W <= 0] = 0.01
    NORM_SAT_W[NORM_SAT_W >= 1] = 0.99
    LOGIT_SAT_W_B, _, LOGIT_SAT_W_1 = get_power1(np.log(NORM_SAT_W / (1 - NORM_SAT_W))
                                 , -0.07596, np.log(0.99 / (1 - 0.99)), 2.7829711, 0.5875367, 'LOGIT_SAT_W')

    LRSPRT_I_B, _, LRSPRT_I_1 = get_power1(np.log(dataset['RSPRT_I'].copy() + 1)
                           , 0.06865, 0, 1.3871886, 0.7812913, 'RSPRT_I')

    LTEMP_I_B, LTEMP_I2_B, _, LTEMP_I_1 = get_power2(np.log((dataset['TEMP_I'].copy() * 1.8 + 0.1))
                                         , 0.020399, 0.14177, np.log(0.1)
                                         , 0.1581225, 1, 0.7654367, 2.0590672, 'LTEMP_I')

    MISS_TROP0_B, MISS_TROP1_B = get_miss_trop(dataset)

    NEURO_B = get_neuro(dataset)

    PML_B, _, PML_1 = get_power1(dataset['PML'].copy()
                      , 0.025628, 300, 317.4747977, 160.7103622, 'PML')

    RSPRT_B, RSPRT2_B, RSPRT3_B, RSPRT_1 = get_power3(dataset['RSPRT'].copy()
                                            , 0.11809, 0.25333, -0.3967, 12.0
                                            , 18.3699826, 1, 1.1748004
                                            , 2.249952, 4.8477904, 87.6749775, 'RSPRT')

    RSPRT_W_B, _, RSPRT_W_1 = get_power1(dataset['RSPRT_W'].copy()
                             , 0.039147, 12, 20.4922021, 3.7967624, 'RSPRT_W')

    SEASON_B = get_season(dataset)

    SEX_B = get_sex(dataset)

    HRTRT = dataset['HRTRT'].copy()
    BPSYS = dataset['BPSYS'].copy()
    HRTRT[np.isnan(HRTRT)] = 80
    BPSYS[np.isnan(BPSYS)] = 110
    BPSYS[BPSYS == 0] = 1
    SHOCK_B, _, SHOCK_1 = get_power1(HRTRT / BPSYS
                            , -0.076924, 80 / 110, 0.6466872, 0.1794605, 'SHOCK')

    SODIUM_B, _, SODIUM_1 = get_power1(dataset['SODIUM']
                            , -0.068568, 140.0, 133.7753625, 10.4324786, 'SODIUM')

    TEMP_B, TEMP2_B, _, TEMP_1 = get_power2(dataset['TEMP'] * 1.8 + 32
                                   , 0.025352, 0.03575, 98.6
                                   , 98.1688675, 1, 0.795785, 2.6491882, 'TEMP')

    TROP_B, _, TROP_1 = get_power1(dataset['TROP'] / 1000
                          , 0.040474, 0, 0.0828239, 0.6741463, 'TROP')

    WBC_B, _, WBC_1 = get_power1(dataset['WBC']
                         , 0.051908, 5, 8.9303325, 4.997061, 'WBC')

    INTERCEPT_B = pd.Series(np.ones(dataset.shape[0]) * -6.88844, name = 'INTERCEPT_B')

    # Missing variables
    COPS2_B = pd.Series(((np.log(11) - 3.2942005) / 1.0162235) * np.ones(dataset.shape[0]) * 0.050426, name = 'COPS2_B')


    # Complete betas
    dataset_beta = pd.concat([ADMIT1_B, ADMIT2_B, ADMIT3_B, ADMIT4_B, ANIONGAP_B, BICARB_B, BICARB2_B, \
                              BPDIA_B, BPDIA2_B, BPSYS_I_B, BPSYS_B, BPSYS2_B, BPSYS3_B, \
                              CARE_ORDERDNR_B, CARE_ORDERFULL_CODE_B, CARE_ORDERPARTIAL_B, \
                              DAYTIME1_B, DAYTIME2_B, DAYTIME3_B, DAYTIME4_B, \
                              GLUCOSE_B, HEMAT_B, HEMAT2_B, HEMAT3_B, \
                              HRTRT_B, HRTRT2_B, HRTRT3_B, \
                              LACT_B, \
                              LAPS2_B, LAPS22_B, LAPS23_B, LAPS2_HET_B, \
                              LBUN_B, LCREAT_B, LCREAT2_B, LELOS_B, ELOSLAPS2_B, \
                              LHRTRT_I_B, LHRTRT2_I_B, LSAT_I_B, \
                              LOGIT_AGE_B, LOGIT_AGE2_B, \
                              LOGIT_SAT_B, LOGIT_SAT2_B, LOGIT_SAT3_B, LOGIT_SAT_W_B, \
                              LRSPRT_I_B, LTEMP_I_B, LTEMP_I2_B, \
                              MISS_TROP0_B, MISS_TROP1_B, NEURO_B, \
                              PML_B, \
                              RSPRT_B, RSPRT2_B, RSPRT3_B, RSPRT_W_B, \
                              SEASON_B, SEX_B, \
                              SHOCK_B, SODIUM_B, TEMP_B, TEMP2_B, TROP_B, WBC_B, \
                              COPS2_B, INTERCEPT_B], axis = 1, ignore_index=False)
    
    # LOGIT sum
    LOGIT = np.sum(dataset_beta.values, axis = 1)

    # ODDS
    ODDS = np.exp(LOGIT)

    # AAM
    AAM = pd.Series((ODDS / (1 + ODDS)) * 100, name = 'AAM')
    if np.any(np.isnan(AAM)):
        pdb.set_trace()
        
    imputed_cols = [
        ANIONGAP_1, BICARB_1, BPDIA_1, BPSYS_I_1, BPSYS_1, GLUCOSE_1,
        HEMAT_1, HRTRT_1, LACT_1, LAPS2_1, LAPS2_HET_1,
        LBUN_1, LCREAT_1, LELOS_1, ELOSLAPS2_1,
        LHRTRT_I_1, LSAT_I_1, LOGIT_AGE_1,
        LOGIT_SAT_1, LOGIT_SAT_W_1, LRSPRT_I_1, LTEMP_I_1,
        PML_1, RSPRT_1, RSPRT_W_1, SHOCK_1,
        SODIUM_1, TEMP_1, TROP_1, WBC_1
    ]
    
    imputed_df = pd.concat(imputed_cols, axis=1)
    

    if 'GLUCOSE_1' in imputed_df.columns:
        imputed_df['GLUCOSE_1'] = imputed_df['GLUCOSE_1'] / 18.02

    if 'HEMAT_1' in imputed_df.columns:
        imputed_df['HEMAT_1'] = imputed_df['HEMAT_1'] / 100

    if 'LBUN_1' in imputed_df.columns:
        imputed_df['LBUN_1'] = (np.exp(imputed_df['LBUN_1']) - 1) / 2.8

    if 'LCREAT_1' in imputed_df.columns:
        imputed_df['LCREAT_1'] = (np.exp(imputed_df['LCREAT_1']) - 0.1) * 88.42

    if 'LELOS_1' in imputed_df.columns:
        imputed_df['LELOS_1'] = np.exp(imputed_df['LELOS_1'])

    if 'LHRTRT_I_1' in imputed_df.columns:
        imputed_df['LHRTRT_I_1'] = np.exp(imputed_df['LHRTRT_I_1']) - 1

    if 'LSAT_I_1' in imputed_df.columns:
        imputed_df['LSAT_I_1'] = np.exp(imputed_df['LSAT_I_1']) - 1

    if 'LOGIT_AGE_1' in imputed_df.columns:
        imputed_df['LOGIT_AGE_1'] = 17 + 100 * (np.exp(imputed_df['LOGIT_AGE_1']) / (1 + np.exp(imputed_df['LOGIT_AGE_1'])))

    if 'LOGIT_SAT_1' in imputed_df.columns:
        imputed_df['LOGIT_SAT_1'] = 1 + 100 * (np.exp(imputed_df['LOGIT_SAT_1']) / (1 + np.exp(imputed_df['LOGIT_SAT_1'])))

    if 'LOGIT_SAT_W_1' in imputed_df.columns:
        imputed_df['LOGIT_SAT_W_1'] = 1 + 100 * (np.exp(imputed_df['LOGIT_SAT_W_1']) / (1 + np.exp(imputed_df['LOGIT_SAT_W_1'])))

    if 'LRSPRT_I_1' in imputed_df.columns:
        imputed_df['LRSPRT_I_1'] = np.exp(imputed_df['LRSPRT_I_1']) - 1

    if 'LTEMP_I_1' in imputed_df.columns:
        imputed_df['LTEMP_I_1'] = (np.exp(imputed_df['LTEMP_I_1']) - 0.1) / 1.8

    if 'TEMP_1' in imputed_df.columns:
        imputed_df['TEMP_1'] = (imputed_df['TEMP_1'] - 32) / 1.8

    if 'TROP_1' in imputed_df.columns:
        imputed_df['TROP_1'] = imputed_df['TROP_1'] * 1000
    
    return AAM, dataset_beta, imputed_df


# Define class
class AAM():
    """AAM model
    Binary classifier using logistic regression.
    The model predicts the risk of patient deterioration for patients on the ward
    
    Parameters:
    t : Threshold, scores above this threshold are classified as at risk of deterioration
    """
    def __init__(self, t):
        self.t = t
    
    def predict(self, dataset):
        """ Prediction function of AAM model
        Get the AAM score for a dataset
        
        Parameters:
        dataset : Should be a pandas dataframe containing the features as columns
        and individual oberservation (by hour) as rows
        """
        proba, _, imputed_df = get_AAM(dataset)
        imp = pd.concat([dataset, imputed_df], axis=1)
        imp['AAM'] = proba
        return imp