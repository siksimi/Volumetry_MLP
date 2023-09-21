# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:54:40 2023

@title: 
Prediction of Mini-Mental State Examination (MMSE) and Seoul Neuropsychological Screening Battery (SNSB) Scores 
Using Brain Volumetry Data From Four Software Packages

@ author: Chae Young Lim, Yongsik Sim, Sohn, Beomseok
@ code: Yongsik Sim

"""

## import libraries ##
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
from shap import GradientExplainer

## codes to ensure reproducibility ##
random_seed = 42

## Load models ##
model_A = keras.models.load_model('SAVED/model_A-20230910-132312.h5')
model_B = keras.models.load_model('SAVED/model_B-20230910-132312.h5')
model_C = keras.models.load_model('SAVED/model_C-20230910-132312.h5')
model_D = keras.models.load_model('SAVED/model_D-20230910-132312.h5')


## import clinical and volumetric data ##
df_clinical = pd.read_csv('data/clinical.csv')
df_clinical['sex'] = df_clinical['sex'].replace(['M', 'F'], [1, 0])
df_volume_A = pd.read_csv('data/volume_aqua.csv')
df_volume_B = pd.read_csv('data/volume_vuno.csv')
df_volume_C = pd.read_csv('data/volume_heuron.csv')
df_volume_D = pd.read_csv('data/volume_fs.csv')


## Training/validation/test set split ##
df_merged = pd.concat([df_clinical, df_volume_A, df_volume_B, df_volume_C, df_volume_D], join='inner', axis=1)
split_time = '2018-08-20 00:00:00 UTC'
df_train_val = df_merged[df_merged['MRI_DATE'] < split_time].drop(columns=['MRI_DATE','ID'])
df_test = df_merged[df_merged['MRI_DATE'] >= split_time].drop(columns=['MRI_DATE','ID'])
df_train, df_val  = train_test_split(df_train_val, test_size=1/8., random_state=random_seed)

print("Training set n = {} ({}%)".format(len(df_train), int(len(df_train)/len(df_merged)*10000)/100))
print("Validation set n = {} ({}%)".format(len(df_val), int(len(df_val)/len(df_merged)*10000)/100))
print("Test set n = {} ({}%)".format(len(df_test), int(len(df_test)/len(df_merged)*10000)/100))


## preparation of input dataframe ##
clinical_variables = ['age', 'edu', 'sex', 'htn', 'dm', 'dys', 'cva']
df_train_clinical = df_train.loc[:, clinical_variables]
df_val_clinical = df_val.loc[:, clinical_variables]
df_test_clinical = df_test.loc[:, clinical_variables]
print ('Number of features in {} : {}'.format('clinical_variables', len(clinical_variables)))

df_train_volume_A = df_train.loc[:, list(df_volume_A.drop(columns=['ID']).columns)]
df_val_volume_A = df_val.loc[:, list(df_volume_A.drop(columns=['ID']).columns)]
df_test_volume_A = df_test.loc[:, list(df_volume_A.drop(columns=['ID']).columns)]
print ('Number of features in {} : {}'.format('df_train_volume_A', df_train_volume_A.shape[1]))

df_train_volume_B = df_train.loc[:, list(df_volume_B.drop(columns=['ID']).columns)]
df_val_volume_B = df_val.loc[:, list(df_volume_B.drop(columns=['ID']).columns)]
df_test_volume_B = df_test.loc[:, list(df_volume_B.drop(columns=['ID']).columns)]
print ('Number of features in {} : {}'.format('df_train_volume_B', df_train_volume_B.shape[1]))

df_train_volume_C = df_train.loc[:, list(df_volume_C.drop(columns=['ID']).columns)]
df_val_volume_C = df_val.loc[:, list(df_volume_C.drop(columns=['ID']).columns)]
df_test_volume_C = df_test.loc[:, list(df_volume_C.drop(columns=['ID']).columns)]
print ('Number of features in {} : {}'.format('df_train_volume_C', df_train_volume_C.shape[1]))

df_train_volume_D = df_train.loc[:, list(df_volume_D.drop(columns=['ID']).columns)]
df_val_volume_D = df_val.loc[:, list(df_volume_D.drop(columns=['ID']).columns)]
df_test_volume_D = df_test.loc[:, list(df_volume_D.drop(columns=['ID']).columns)]
print ('Number of features in {} : {}'.format('df_train_volume_D', df_train_volume_D.shape[1]))


## preparation of ground truth output dataframe ##
output_variables = ['mmse', 'attention', 'language', 'visuospatial', 'memory', 'frontal']
df_train_output = df_train.loc[:, output_variables]
df_val_output = df_val.loc[:, output_variables]
df_test_output = df_test.loc[:, output_variables]
print ('Number of features in {} : {}'.format('output_variables', len(output_variables)))


## standarscaler ##
def fit_transform_scaler(df_train):
    scaler = StandardScaler()
    df_train_to_return = scaler.fit_transform(df_train)
    return(df_train_to_return, scaler)
def transform_scaler(scaler, df_val, df_test):
    df_val_to_return = scaler.transform(df_val)
    df_test_to_return = scaler.transform(df_test)
    return(df_val_to_return, df_test_to_return)

df_train_clinical, clinical_scaler = fit_transform_scaler(df_train_clinical)
df_train_volume_A, A_scaler = fit_transform_scaler(df_train_volume_A)
df_train_volume_B, B_scaler = fit_transform_scaler(df_train_volume_B)
df_train_volume_C, C_scaler = fit_transform_scaler(df_train_volume_C)
df_train_volume_D, D_scaler = fit_transform_scaler(df_train_volume_D)

df_val_clinical, df_test_clinical = transform_scaler(clinical_scaler, df_val_clinical, df_test_clinical)
df_val_volume_A, df_test_volume_A = transform_scaler(A_scaler, df_val_volume_A, df_test_volume_A)
df_val_volume_B, df_test_volume_B = transform_scaler(B_scaler, df_val_volume_B, df_test_volume_B)
df_val_volume_C, df_test_volume_C = transform_scaler(C_scaler, df_val_volume_C, df_test_volume_C)
df_val_volume_D, df_test_volume_D = transform_scaler(D_scaler, df_val_volume_D, df_test_volume_D)

## SHAP analysis ##
def run_SHAP(model, df_vol, df_train_vol, model_name):
    explainer = GradientExplainer(model, [df_train_clinical, df_train_vol])
    shap_values = explainer.shap_values([df_train_clinical, df_train_vol])
    
    for n, var in enumerate (output_variables):
        shap_value = shap_values[n]
        shap_clinical = shap_value[0]
        shap_volume = shap_value[1]
        result = []
        for j in range (7):
            label = clinical_variables[j]
            sum = 0
            for i in range (len(shap_clinical)):
                sum += abs(shap_clinical[i][j])
            avg = sum/len(shap_clinical)
            result.append([label, avg])
        df_csv_clinical = pd.DataFrame(result, columns =["Clinical data", "mean(abs(SHAP))"]).sort_values("mean(abs(SHAP))", ascending=False)
        df_csv_clinical.to_csv("result_shap/{}_{}_clinical.csv".format(model_name, var))
        result2 = []
        for j in range (len(shap_volume[0])):
            label = df_vol.drop(columns=['ID']).columns[j]
            sum = 0
            for i in range (len(shap_volume)):
                sum += abs(shap_volume[i][j])
            avg = sum/len(shap_volume)
            result2.append([label, avg])  
        df_csv_volume = pd.DataFrame(result2, columns =["volume_ID", "mean(abs(SHAP))"]).sort_values("mean(abs(SHAP))", ascending=False)
        df_csv_volume.to_csv("result_shap/{}_{}_volume.csv".format(model_name, var))    
        
run_SHAP(model_A, df_volume_A, df_train_volume_A, "model_A")
run_SHAP(model_B, df_volume_B, df_train_volume_B, "model_B")
run_SHAP(model_C, df_volume_C, df_train_volume_C, "model_C")
run_SHAP(model_D, df_volume_D, df_train_volume_D, "model_D")        