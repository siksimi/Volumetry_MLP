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
import os
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping

## codes to ensure reproducibility ##
random_seed = 42
os.environ["PYTHONHASHSEED"] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
keras.utils.set_random_seed(random_seed)
tf.config.experimental.enable_op_determinism()


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


## data augmentation ##
train_columns = df_train.columns
df_train_copy = np.tile(df_train, (20, 1))
mu, sigma = 1, 0.04 # mean and standard deviation
s = np.random.normal(mu, sigma, df_train_copy.shape[0]*df_train_copy.shape[1])
count, bins, ignored = plt.hist(s, 30, density=True)
df_noise = s.reshape(df_train_copy.shape)
df_train =pd.DataFrame(np.multiply(df_train_copy,df_noise), columns= train_columns)


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


## model construction ##
def model_construct(No_clinical_variables, No_volume_variables, model_name):
    input_clinical = Input(shape = No_clinical_variables, name = "clinical_input")
    input_volume = Input(shape = No_volume_variables, name = "volume_input")
    hidden = Dense (60, activation = "relu")(input_volume)
    concat = concatenate([input_clinical, hidden])
    hidden2 = Dense (30, activation = "relu")(concat)
    dropout2 = Dropout(rate=0.2)(hidden2)
    output = Dense (len(output_variables), name="output")(dropout2)
    model = keras.Model(inputs=[input_clinical, input_volume], outputs=[output], name=model_name)
    model.summary()
    return model

model_A = model_construct(len(clinical_variables), df_train_volume_A.shape[1], "model_A")
model_B = model_construct(len(clinical_variables), df_train_volume_B.shape[1], "model_B")
model_C = model_construct(len(clinical_variables), df_train_volume_C.shape[1], "model_C")
model_D = model_construct(len(clinical_variables), df_train_volume_D.shape[1], "model_D")


## model training ##
def model_train(model, df_clinical, df_volume, df_output, validation_data):
    logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=2e-3))
    return model.fit((df_clinical, df_volume), df_output, epochs= 1000, batch_size = 256, 
                     validation_data=validation_data,
                     callbacks=[early_stopping_callback, tensorboard_callback])

history_A = model_train(model_A, df_train_clinical, df_train_volume_A, df_train_output, ((df_val_clinical, df_val_volume_A),df_val_output))
history_B = model_train(model_B, df_train_clinical, df_train_volume_B, df_train_output, ((df_val_clinical, df_val_volume_B),df_val_output))
history_C = model_train(model_C, df_train_clinical, df_train_volume_C, df_train_output, ((df_val_clinical, df_val_volume_C),df_val_output))
history_D = model_train(model_D, df_train_clinical, df_train_volume_D, df_train_output, ((df_val_clinical, df_val_volume_D),df_val_output))


## model evaluation ##
def model_eval(model, df_clinical, df_volume, df_output):
    y_pred = model.predict((df_clinical, df_volume))
    df_pred = pd.DataFrame(y_pred, columns = output_variables)
    df_pred['mmse'].values[df_pred['mmse'] > 30] = 30
    result = np.empty((0,5))
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    for i, var in enumerate(output_variables):
        r, p = stats.pearsonr(df_output[var], df_pred[var])
        sns.regplot(x=df_output[var], y=df_pred[var], ax=axs[int(i/3), i%3])
        result = np.append(result, np.array([[var, round(mean_absolute_error(df_output[var], df_pred[var]),4), round(mean_squared_error(df_output[var], df_pred[var]),4), round(r, 4), round(r**2,4)]]), axis=0)
    df_result = pd.DataFrame(result, columns = ["variable", "mae", "mse", "Pearson R", "R square"])
    print(df_result)
    plt.show()
    return df_pred, result

pred_test_A = model_eval(model_A, df_test_clinical, df_test_volume_A, df_test_output)
pred_test_B = model_eval(model_B, df_test_clinical, df_test_volume_B, df_test_output)
pred_test_C = model_eval(model_C, df_test_clinical, df_test_volume_C, df_test_output)
pred_test_D = model_eval(model_D, df_test_clinical, df_test_volume_D, df_test_output)


## save prediction & true data ##
pred_test_A[0].to_csv("prediction_save/A_prediction.csv")
pred_test_B[0].to_csv("prediction_save/B_prediction.csv")
pred_test_C[0].to_csv("prediction_save/C_prediction.csv")
pred_test_D[0].to_csv("prediction_save/D_prediction.csv")
df_test_output.to_csv("true_save/Test_true.csv")

## save model ##
model_A.save('model_save/model_A-{}.h5'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))
model_B.save('model_save/model_B-{}.h5'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))
model_C.save('model_save/model_C-{}.h5'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))
model_D.save('model_save/model_D-{}.h5'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))