#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday January 11, 11:38:46, 2022

@author: omaranwar
"""

#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

import sklearn
#from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.inspection import permutation_importance




# Create lists to store the scores
MAE_list = [0] * 13
MAE_index= 0
MAE_all = []

#load the features from numpy file as a numpy array
AllFeatures = np.load('AllFeatures_shuffled_144.npy',allow_pickle=True)


# Rearrange
AllFeaturesN=AllFeatures.transpose(0, 2, 1)
m,n,r = AllFeaturesN.shape

# Create an empty list with size = total number of days in dataset
RF_list = [ [] for _ in range(m)]
y_test_all = []
y_pred_all = []

# Populate the list with features
for kk in range(m):
    # Information about week/season is encoded as a pair of sin and cos waveforms, with a period of 52.14 (weeks in one year). 
    # First two column features have time information encoded as Sin, Cos with 1,440 minute period, which is not used here. AllFeaturesN[kk,0,0] and AllFeaturesN[kk,0,1]
    RF_list[kk].append(AllFeaturesN[kk,0,2]) # WeekSin
    RF_list[kk].append(AllFeaturesN[kk,0,3]) # WeekCos
    for jj in range(4,35): # For all the features starting from temp inside the hive -> Z-axis of accelerometer. A total of 31 features are populates in this loop
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[0:18].mean()) #mean of 18 samples, corresponding to 3 hour windows
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[18:36].mean())
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[36:54].mean())
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[54:72].mean())
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[72:90].mean())
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[90:108].mean())
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[108:126].mean())
        RF_list[kk].append((AllFeaturesN[kk,:,jj])[126:144].mean())

    Hive_frames = AllFeaturesN[kk,143,35] 
    RF_list[kk].append(Hive_frames) # number of frames in the hive
    RF_list[kk].append(float(AllFeaturesN[kk,143,36]*Hive_frames)) #Weight change per hive frame multiplied by total number of frames = total weight change of hive per day
    RF_list[kk].append(AllFeaturesN[kk,143,37]*Hive_frames) #Weight per hive frame at the start of day multiplied by total number of frames = total weight of hive at start of day. 
                                                            #This is only for reference and is not used in weight estimation below
    RF_list[kk].append(AllFeaturesN[kk,143,38]) # ID assigned to the hive, not used in weight estimation below
    RF_list[kk].append(AllFeaturesN[kk,143,39]) # Date of data collection, not used in weight estimation below
#This generates a list of lists with 255 elements per list, corresponding to the data for a single day per hive
    

column_names = [ # names of all the columns in dataframe
            'WeekSin','WeekCos',
            'T_in_01','T_in_04','T_in_07','T_in_10','T_in_13','T_in_16','T_in_19','T_in_22',
            'T_in_grad_01','T_in_grad_04','T_in_grad_07','T_in_grad_10','T_in_grad_13','T_in_grad_16','T_in_grad_19','T_in_grad_22',
            'T_out_01','T_out_04','T_out_07','T_out_10','T_out_13','T_out_16','T_out_19','T_out_22',
            'T_out_feel_01','T_out_feel_04','T_out_feel_07','T_out_feel_10','T_out_feel_13','T_out_feel_16','T_out_feel_19','T_out_feel_22',
            'T_diff_01','T_diff_04','T_diff_07','T_diff_10','T_diff_13','T_diff_16','T_diff_19','T_diff_22',
            'HumidIn_01','HumidIn_04','HumidIn_07','HumidIn_10','Humidin_13','Humidin_16','Humidin_19','Humidin_22',
            'HumidOut_01','HumidOut_04','HumidOut_07','HumidOut_10','HumidOut_13','HumidOut_16','HumidOut_19','HumidOut_22',
            'Wind_01','Wind_04','Wind_07','Wind_10','Wind_13','Wind_16','Wind_19','Wind_22',
            'rain_01','rain_04','rain_07','rain_10','rain_13','rain_16','rain_19','rain_22',
            'CO2_01','CO2_04','CO2_07','CO2_10','CO2_13','CO2_16','CO2_19','CO2_22',
            'Pr_01','Pr_04','Pr_07','Pr_10','Pr_13','Pr_16','Pr_19','Pr_22',
            'AudFreq_01','AudFreq_04','AudFreq_07','AudFreq_10','AudFreq_13','AudFreq_16','AudFreq_19','AudFreq_22',
            'AudAmp_01','AudAmp_04','AudAmp_07','AudAmp_10','AudAmp_13','AudAmp_16','AudAmp_19','AudAmp_22',
            'Band1_01','Band1_04','Band1_07','Band1_10','Band1_13','Band1_16','Band1_19','Band1_22',
            'Band2_01','Band2_04','Band2_07','Band2_10','Band2_13','Band2_16','Band2_19','Band2_22',
            'Band3_01','Band3_04','Band3_07','Band3_10','Band3_13','Band3_16','Band3_19','Band3_22',
            'Band4_01','Band4_04','Band4_07','Band4_10','Band4_13','Band4_16','Band4_19','Band4_22',
            'Band5_01','Band5_04','Band5_07','Band5_10','Band5_13','Band5_16','Band5_19','Band5_22',
            'Band6_01','Band6_04','Band6_07','Band6_10','Band6_13','Band6_16','Band6_19','Band6_22',
            'Band7_01','Band7_04','Band7_07','Band7_10','Band7_13','Band7_16','Band7_19','Band7_22',
            'Band8_01','Band8_04','Band8_07','Band8_10','Band8_13','Band8_16','Band8_19','Band8_22',
            'Band9_01','Band9_04','Band9_07','Band9_10','Band9_13','Band9_16','Band9_19','Band9_22',
            'Band10_01','Band10_04','Band10_07','Band10_10','Band10_13','Band10_16','Band10_19','Band10_22',
            'Band11_01','Band11_04','Band11_07','Band11_10','Band11_13','Band11_16','Band11_19','Band11_22',
            'Band12_01','Band12_04','Band12_07','Band12_10','Band12_13','Band12_16','Band12_19','Band12_22',
            'Band13_01','Band13_04','Band13_07','Band13_10','Band13_13','Band13_16','Band13_19','Band13_22',
            'Band14_01','Band14_04','Band14_07','Band14_10','Band14_13','Band14_16','Band14_19','Band14_22',
            'Band15_01','Band15_04','Band15_07','Band15_10','Band15_13','Band15_16','Band15_19','Band15_22',
            'AccelX_01','AccelX_04','AccelX_07','AccelX_10','AccelX_13','AccelX_16','AccelX_19','AccelX_22',
            'AccelY_01','AccelY_04','AccelY_07','AccelY_10','AccelY_13','AccelY_16','AccelY_19','AccelY_22',
            'AccelZ_01','AccelZ_04','AccelZ_07','AccelZ_10','AccelZ_13','AccelZ_16','AccelZ_19','AccelZ_22',
            'Frames',
            'Weight_var', 'Weight_base', 'ID', 'Date']

Feature_All = [ # names of all the features used for weight estimation
            'WeekSin','WeekCos',
            'T_in_01','T_in_04','T_in_07','T_in_10','T_in_13','T_in_16','T_in_19','T_in_22',
            'T_in_grad_01','T_in_grad_04','T_in_grad_07','T_in_grad_10','T_in_grad_13','T_in_grad_16','T_in_grad_19','T_in_grad_22',
            'T_out_01','T_out_04','T_out_07','T_out_10','T_out_13','T_out_16','T_out_19','T_out_22',
            'T_out_feel_01','T_out_feel_04','T_out_feel_07','T_out_feel_10','T_out_feel_13','T_out_feel_16','T_out_feel_19','T_out_feel_22',
            'T_diff_01','T_diff_04','T_diff_07','T_diff_10','T_diff_13','T_diff_16','T_diff_19','T_diff_22',
            'HumidIn_01','HumidIn_04','HumidIn_07','HumidIn_10','Humidin_13','Humidin_16','Humidin_19','Humidin_22',
            'HumidOut_01','HumidOut_04','HumidOut_07','HumidOut_10','HumidOut_13','HumidOut_16','HumidOut_19','HumidOut_22',
            'Wind_01','Wind_04','Wind_07','Wind_10','Wind_13','Wind_16','Wind_19','Wind_22',
            'rain_01','rain_04','rain_07','rain_10','rain_13','rain_16','rain_19','rain_22',
            'CO2_01','CO2_04','CO2_07','CO2_10','CO2_13','CO2_16','CO2_19','CO2_22',
            'Pr_01','Pr_04','Pr_07','Pr_10','Pr_13','Pr_16','Pr_19','Pr_22',
            'AudFreq_01','AudFreq_04','AudFreq_07','AudFreq_10','AudFreq_13','AudFreq_16','AudFreq_19','AudFreq_22',
            'AudAmp_01','AudAmp_04','AudAmp_07','AudAmp_10','AudAmp_13','AudAmp_16','AudAmp_19','AudAmp_22',
            'Band1_01','Band1_04','Band1_07','Band1_10','Band1_13','Band1_16','Band1_19','Band1_22',
            'Band2_01','Band2_04','Band2_07','Band2_10','Band2_13','Band2_16','Band2_19','Band2_22',
            'Band3_01','Band3_04','Band3_07','Band3_10','Band3_13','Band3_16','Band3_19','Band3_22',
            'Band4_01','Band4_04','Band4_07','Band4_10','Band4_13','Band4_16','Band4_19','Band4_22',
            'Band5_01','Band5_04','Band5_07','Band5_10','Band5_13','Band5_16','Band5_19','Band5_22',
            'Band6_01','Band6_04','Band6_07','Band6_10','Band6_13','Band6_16','Band6_19','Band6_22',
            'Band7_01','Band7_04','Band7_07','Band7_10','Band7_13','Band7_16','Band7_19','Band7_22',
            'Band8_01','Band8_04','Band8_07','Band8_10','Band8_13','Band8_16','Band8_19','Band8_22',
            'Band9_01','Band9_04','Band9_07','Band9_10','Band9_13','Band9_16','Band9_19','Band9_22',
            'Band10_01','Band10_04','Band10_07','Band10_10','Band10_13','Band10_16','Band10_19','Band10_22',
            'Band11_01','Band11_04','Band11_07','Band11_10','Band11_13','Band11_16','Band11_19','Band11_22',
            'Band12_01','Band12_04','Band12_07','Band12_10','Band12_13','Band12_16','Band12_19','Band12_22',
            'Band13_01','Band13_04','Band13_07','Band13_10','Band13_13','Band13_16','Band13_19','Band13_22',
            'Band14_01','Band14_04','Band14_07','Band14_10','Band14_13','Band14_16','Band14_19','Band14_22',
            'Band15_01','Band15_04','Band15_07','Band15_10','Band15_13','Band15_16','Band15_19','Band15_22',
            'AccelX_01','AccelX_04','AccelX_07','AccelX_10','AccelX_13','AccelX_16','AccelX_19','AccelX_22',
            'AccelY_01','AccelY_04','AccelY_07','AccelY_10','AccelY_13','AccelY_16','AccelY_19','AccelY_22',
            'AccelZ_01','AccelZ_04','AccelZ_07','AccelZ_10','AccelZ_13','AccelZ_16','AccelZ_19','AccelZ_22',
            'Frames'
            ]

Feature_Season = [
            'WeekSin','WeekCos',
            ]
Feature_Temp = [
            'T_in_01','T_in_04','T_in_07','T_in_10','T_in_13','T_in_16','T_in_19','T_in_22',
            'T_in_grad_01','T_in_grad_04','T_in_grad_07','T_in_grad_10','T_in_grad_13','T_in_grad_16','T_in_grad_19','T_in_grad_22',
            ]
Feature_Humidity = [          
            'HumidIn_01','HumidIn_04','HumidIn_07','HumidIn_10','Humidin_13','Humidin_16','Humidin_19','Humidin_22',
            ]
Feature_CO2 = [
            'CO2_01','CO2_04','CO2_07','CO2_10','CO2_13','CO2_16','CO2_19','CO2_22',
            ]
Feature_Pressure = [
            'Pr_01','Pr_04','Pr_07','Pr_10','Pr_13','Pr_16','Pr_19','Pr_22',
            ]
Feature_Audio = [
            'AudFreq_01','AudFreq_04','AudFreq_07','AudFreq_10','AudFreq_13','AudFreq_16','AudFreq_19','AudFreq_22',
            'AudAmp_01','AudAmp_04','AudAmp_07','AudAmp_10','AudAmp_13','AudAmp_16','AudAmp_19','AudAmp_22',
            'Band1_01','Band1_04','Band1_07','Band1_10','Band1_13','Band1_16','Band1_19','Band1_22',
            'Band2_01','Band2_04','Band2_07','Band2_10','Band2_13','Band2_16','Band2_19','Band2_22',
            'Band3_01','Band3_04','Band3_07','Band3_10','Band3_13','Band3_16','Band3_19','Band3_22',
            'Band4_01','Band4_04','Band4_07','Band4_10','Band4_13','Band4_16','Band4_19','Band4_22',
            'Band5_01','Band5_04','Band5_07','Band5_10','Band5_13','Band5_16','Band5_19','Band5_22',
            'Band6_01','Band6_04','Band6_07','Band6_10','Band6_13','Band6_16','Band6_19','Band6_22',
            'Band7_01','Band7_04','Band7_07','Band7_10','Band7_13','Band7_16','Band7_19','Band7_22',
            'Band8_01','Band8_04','Band8_07','Band8_10','Band8_13','Band8_16','Band8_19','Band8_22',
            'Band9_01','Band9_04','Band9_07','Band9_10','Band9_13','Band9_16','Band9_19','Band9_22',
            'Band10_01','Band10_04','Band10_07','Band10_10','Band10_13','Band10_16','Band10_19','Band10_22',
            'Band11_01','Band11_04','Band11_07','Band11_10','Band11_13','Band11_16','Band11_19','Band11_22',
            'Band12_01','Band12_04','Band12_07','Band12_10','Band12_13','Band12_16','Band12_19','Band12_22',
            'Band13_01','Band13_04','Band13_07','Band13_10','Band13_13','Band13_16','Band13_19','Band13_22',
            'Band14_01','Band14_04','Band14_07','Band14_10','Band14_13','Band14_16','Band14_19','Band14_22',
            'Band15_01','Band15_04','Band15_07','Band15_10','Band15_13','Band15_16','Band15_19','Band15_22',
            ]
Feature_Accel = [
            'AccelX_01','AccelX_04','AccelX_07','AccelX_10','AccelX_13','AccelX_16','AccelX_19','AccelX_22',
            'AccelY_01','AccelY_04','AccelY_07','AccelY_10','AccelY_13','AccelY_16','AccelY_19','AccelY_22',
            'AccelZ_01','AccelZ_04','AccelZ_07','AccelZ_10','AccelZ_13','AccelZ_16','AccelZ_19','AccelZ_22',
            ]

Feature_Temp_out = [
            'T_out_01','T_out_04','T_out_07','T_out_10','T_out_13','T_out_16','T_out_19','T_out_22',
            ]
Feature_Temp_out_feel = [
            'T_out_feel_01','T_out_feel_04','T_out_feel_07','T_out_feel_10','T_out_feel_13','T_out_feel_16','T_out_feel_19','T_out_feel_22',
            ]
Feature_Humid_out = [
            'HumidOut_01','HumidOut_04','HumidOut_07','HumidOut_10','HumidOut_13','HumidOut_16','HumidOut_19','HumidOut_22',
            ]
Feature_rain = [
            'rain_01','rain_04','rain_07','rain_10','rain_13','rain_16','rain_19','rain_22',
            ]
Feature_wind = [
            'Wind_01','Wind_04','Wind_07','Wind_10','Wind_13','Wind_16','Wind_19','Wind_22',
            ]


Features_list=[Feature_All,Feature_Audio,Feature_Temp,Feature_Season,Feature_Pressure,Feature_Humidity,Feature_CO2,
               Feature_Accel,Feature_Humid_out,Feature_Temp_out,Feature_Temp_out_feel,Feature_wind,Feature_rain] # list of all the feature lists to be tested

Features_list_names=['All Features','Audio','Temperature','Season','Pressure','Humidity','CO2','Accelero', # names of all the feature lists to be tested, used in plots.
                      'Humid_out', 'Temp_out', 'Temp_out_feel', 'Wind', 'Rain']

RF_df_all = pd.DataFrame.from_records(RF_list,columns=column_names) #convert list to data frame 

df_size = RF_df_all.shape[0]

RF_df = RF_df_all

#iterate through all the lists eand evaluate features for their effectiveness for daily weight change estimation
for k in range(len(Features_list)):
    
    Feature_names = Features_list[k]
    Feature_names_string = Features_list_names[k]
    
    print('---------------------------------------------')
    print('Processing for following features: ',Feature_names)
    Importances = []
    y_test_all = []
    y_pred_all = []
        
    k_fold = 5
    for fold in range(k_fold):
        split_offset = fold * (1/k_fold)

        test_start = int(round(df_size*(0+split_offset)))
        test_stop = int(round(df_size*((1/k_fold)+split_offset)))
        print('Fitting the model for fold:',fold)
       
    
        df_split_A = RF_df.iloc[0:test_start, :]
        df_split_B = RF_df.iloc[test_start:test_stop, :]
        df_split_C = RF_df.iloc[test_stop:, :]
    
        pd_frames = [df_split_A,  df_split_C]
        train_df_orig = pd.concat(pd_frames)

        test_df_orig = df_split_B

        
        X_train = train_df_orig.loc[:, Feature_names]
        y_train = train_df_orig.Weight_var
        
        # X_test = X_train
        # y_test = y_train
        X_test = test_df_orig.loc[:, Feature_names]
        y_test = test_df_orig.Weight_var
        
        random_forest = RandomForestRegressor(n_estimators = 100, max_depth=len(Feature_names), oob_score=True, max_samples = 800)
        if(fold == 0): # search for best params only on the first fold, and use the same params for all folds for a given set of features
            SEED = 13
            # random_forest_tuning = RandomForestRegressor(random_state = SEED)
            # param_grid = {
            #     # 'criterion' : ['squared_error', 'absolute_error', 'poisson'],
            #     'n_estimators': [50,100,150],
            #     'max_features': ['auto','sqrt'],
            #     'max_depth' : [len(Feature_names), len(Feature_names)/2],
            #     'max_samples' :[400, 600, 800]
            # }
            # random_forest = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5)
        
            random_forest.fit(X_train, y_train)
            #print(random_forest.best_params_)
        
        else:
            random_forest.fit(X_train, y_train)
        print('Predicting values for testset and evaluating feature importances.')
        print('test_start:', test_start)
        print('test_stop:', test_stop)
       
        y_pred = random_forest.predict(X_test) 
        
        MAE = mean_absolute_error(y_test, y_pred)
        #print('MAE: ', MAE)
        MAE_all.append(MAE)
        #print('MSE: ', mean_squared_error(y_test, y_pred))
        
        MAE_list[MAE_index]=MAE_list[MAE_index]+(mean_absolute_error(y_test, y_pred))/k_fold
        y_test_all = y_test_all + y_test.tolist()
        y_pred_all = y_pred_all + y_pred.tolist()
        
        
        start_time = time.time()
        result = permutation_importance(
            random_forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=5
        )
        elapsed_time = time.time() - start_time
        #print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
        if(fold == 0):
            Importances = result.importances_mean/k_fold
        else:
            Importances = Importances + result.importances_mean/k_fold
        forest_importances = pd.Series(result.importances_mean, index=Feature_names)
        
    
        print('----------------')
    #generate scatter plots    
    colors = np.random.random((len(y_test_all),3))
    markers =(['o']*1250)[:len(y_test_all)]
    data = {'Actual weight change per hive (kg)': y_test_all, 'Estimated weight change per hive (kg)': y_pred_all}  
    df_2 = pd.DataFrame(data) 
    
    g=sns.jointplot(x="Actual weight change per hive (kg)", y="Estimated weight change per hive (kg)", data=df_2, kind="reg", xlim=(-1, 2.5), ylim=(-1, 2.5))
    
    
    g.ax_joint.collections[0].set_visible(False)
    
    #Plot each individual point separately
    for i,row in enumerate(df_2.values):
        g.ax_joint.plot(row[0], row[1], color=colors[i], marker=markers[i])
     
    g.ax_joint.set_title(Feature_names_string)
    # print('MAE for 5 folds:')
    # print(MAE_all)
    
    
    #generate plot for feature importances
    print('Average MAE for',Feature_names_string)
    print(MAE_list[MAE_index])
    
    forest_importances = pd.Series(Importances, index=Feature_names)
        
    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    #forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.bar(Feature_names,Importances)
    ax.set_title("Average of Feature importances using permutation on full model (all folds) for "+Feature_names_string)
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
    MAE_index = MAE_index + 1

plt.figure()
plt.plot(Features_list_names,MAE_list)
plt.xticks(rotation=90)
plt.title("MAE (Mean Absolute error) for all 13 set of features.")
plt.ylabel("MAE")

