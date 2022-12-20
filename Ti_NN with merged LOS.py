#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:31:06 2021

@author: debsel
"""


# %% - #Importing python tools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow.keras.utils import plot_model
#from tensorflow.keras.optimizers import Adam
from keras.optimizers import adam_v2

print('Libraries for analysis loaded successfully')
print('-----------------')

# %% - #For printing on plots

# Function to convert to subscript
def get_sub(x):
	normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
	sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
	res = x.maketrans(''.join(normal), ''.join(sub_s))
	return x.translate(res)

# Function to convert to superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)
  
print('Functions to print sub & super scripts loaded successfully')
print('-----------------')

# %% - #Importing routines to read data
import sys
sys.path.append('/afs/ipp/home/d/debsel/SpectraViewer/')

from SynSignal import *
sys.path.insert(1, '/afs/ipp/home/d/debsel/SpectraViewer/fidasim')
import read_data_ML

#to read spectra for each shot
readS_1 = read_data_ML.read_spec_1 
readS_2 = read_data_ML.read_spec_2 
readS_3 = read_data_ML.read_spec_3 

#to read profiles for each shot
readP_1 = read_data_ML.read_profiles_1 
readP_2 = read_data_ML.read_profiles_2 
readP_3 = read_data_ML.read_profiles_3 

print('Routines for reading data loaded successfully')
print('-----------------')

# %% - #Reading Spectra Data (selecting ti scans LScan1-9 for each shot)

no_ti_scans = 4
no_scans = 9
ti_scan = ['0.25','0.50','1.00','3.00']*no_scans #multiply by number of elements in scan_number #to load scan ti 0.25-5.00
scan_number = ['1','2', '3','4', '5','6','7','8','9']*no_ti_scans #multiply by number of elements in ti_scan #to load scans LScan1-9

number = no_ti_scans*no_scans #count number of scans to be loaded per shot

shots = ['38587','35840','38539']
n_shots = len(shots)

total_number = n_shots*number

print('Total number of shots to be loaded is: ', total_number)
print(' ')

#Loading data
#for spectra
S_data_1 = [ [] for _ in range(number) ] #for shot 1 #list of empty lists
S_data_2 = [ [] for _ in range(number) ] #for shot 2 
S_data_3 = [ [] for _ in range(number) ] #for shot 3 

Empties = []

for i in range(number):
    try:
        #Loading spectra bin
        S_data_1[i] = readS_1('BOS_38587_LScan%s_ti_%s'%(scan_number[i],ti_scan[i]))
        S_data_2[i] = readS_2('BOS_38540_LScan%s_ti_%s'%(scan_number[i],ti_scan[i])) 
        S_data_3[i] = readS_3('BOS_38539_LScan%s_ti_%s'%(scan_number[i],ti_scan[i])) 
        
    except FileNotFoundError:
        print('A file was not found')
        Empties.append(i)
        print('Data point %s to be removed from loaded list'%(i))
        pass   

#Removing empty datapoints
for i in range(len(Empties)):
    #shot 1
    S_data_1.remove([])
    #shot 2
    S_data_2.remove([])
    #shot 3
    S_data_3.remove([])

New_number = len(S_data_1)
print('-------------------------------')
print('Final number of dataset being used per shot ', New_number)

# %% - #Reading Profile Data

#Test Profile load with SynSignal
#Reading Profile Data (selecting ti scans LScan1-9 for each shot)

no_ti_scans = 4
no_scans = 9
ti_scan = ['0.25','0.50','1.00','3.00']*no_scans #multiply by number of elements in scan_number #to load scan ti0.25-5.00
scan_number = ['1','2', '3','4', '5','6','7','8','9']*no_ti_scans #multiply by number of elements in ti_scan #to load scans LScan1-9

number = no_ti_scans*no_scans #count number of scans to be loaded
shots = ['38587','35840','38539']
n_shots = len(shots)
total_number = n_shots*number

print('Total number of shots to be loaded is: ', total_number)
print(' ')

#for profiles
root1 = '/afs/ipp/home/d/debsel/FIDASIM4/RESULTS/Scans/Linear/38587/'
root2 = '/afs/ipp/home/d/debsel/FIDASIM4/RESULTS/Scans/Linear/38540/'
root3 = '/afs/ipp/home/d/debsel/FIDASIM4/RESULTS/Scans/Linear/38539/'

id_1 = [ [] for _ in range(number) ] 
id_2 = [ [] for _ in range(number) ] 
id_3 = [ [] for _ in range(number) ] 

P_data_1 = [ [] for _ in range(number) ] 
P_data_2 = [ [] for _ in range(number) ] 
P_data_3 = [ [] for _ in range(number) ] 

Empties = []

##Example of routine
# root = '/afs/ipp/home/d/debsel/FIDASIM4/RESULTS/Scans/Linear/38539_old/'
# fidasim_id = 'BOS_38539_LScan3_ti_1.00' #'38876_1559ms'#
# test_synSig = SynSignal(root+fidasim_id)
# rho_chan  = test_synSig.rhop_los
# ti_chan =  np.interp(rho_chan, test_synSig.profFIDASIM['rhop'],test_synSig.profFIDASIM['ti'])


for i in range(number):
    try:
        #make id for each file
        #Profiles
        id_1[i] = 'BOS_38587_LScan%s_ti_%s'%(scan_number[i],ti_scan[i])
        id_2[i] = 'BOS_38540_LScan%s_ti_%s'%(scan_number[i],ti_scan[i])
        id_3[i] = 'BOS_38539_LScan%s_ti_%s'%(scan_number[i],ti_scan[i])
        
        P_data_1[i] = SynSignal(root1 + id_1[i])
        P_data_2[i] = SynSignal(root2 + id_2[i])
        P_data_3[i] = SynSignal(root3 + id_3[i])
        
    except FileNotFoundError:
        print('A file was not found')
        Empties.append(i)
        print('Data point %s to be removed from loaded list'%(i))
        pass

Prof_number = len(P_data_1)
print('-------------------------------')
print('Final number of dataset being used per shot ', Prof_number)
# %%  - #Selecting components

#shot 1
#for spectra
S_x_1 = [ [] for _ in range(New_number) ] #sets of lambda
S_y_1 = [ [] for _ in range(New_number) ] #sets of spectra
#for temp profiles
T_x_1 = [ [] for _ in range(New_number) ] #sets of rho
T_y_1 = [ [] for _ in range(New_number) ] #sets of temp profile values

#shot 2
#for spectra
S_x_2 = [ [] for _ in range(New_number) ] #sets of lambda
S_y_2 = [ [] for _ in range(New_number) ] #sets of spectra
#for temp profiles
T_x_2 = [ [] for _ in range(New_number) ] #sets of rho
T_y_2 = [ [] for _ in range(New_number) ] #sets of temp profile values

#shot 3
#for spectra
S_x_3 = [ [] for _ in range(New_number) ] #sets of lambda
S_y_3 = [ [] for _ in range(New_number) ] #sets of spectra
#for temp profiles
T_x_3 = [ [] for _ in range(New_number) ] #sets of rho
T_y_3 = [ [] for _ in range(New_number) ] #sets of temp profile values


#Selecting components
for i in range(New_number):
    #shot 1
    #sorting rho_pol values
    irho_1 = np.argsort(P_data_1[i].rhop_los) 
    #loading profiles
    T_x_1[i] = P_data_1[i].rhop_los[irho_1] #rho_pol
    T_y_1[i] = np.interp(T_x_1[i][irho_1], P_data_1[i].profFIDASIM['rhop'], P_data_1[i].profFIDASIM['ti']) #ti values
    #loading spectra
    S_x_1[i] = S_data_1[i]['lambda'] #wavelength
    S_x_1[i] = S_x_1[i][780:1050] #removing invalid wavelengths
    S_y_1[i] = S_data_1[i]['halo'] #halo values #shape = (20,2000,15)
    S_y_1[i] = np.sum(S_y_1[i],axis=2) # summing nstark #shape = (20,2000)
    S_y_1[i] = S_y_1[i][irho_1,780:1050] #sorting spectra and removing values of invalid wavelengths 
    S_y_1[i] = np.array(S_y_1[i]).flatten() #merging data from all LOS

    #shot 2
    #sorting rho_pol values
    irho_2 = np.argsort(P_data_2[i].rhop_los)
    #loading profiles
    T_x_2[i] = P_data_2[i].rhop_los[irho_2] #rho_pol
    T_y_2[i] = np.interp(T_x_2[i][irho_2], P_data_2[i].profFIDASIM['rhop'], P_data_2[i].profFIDASIM['ti']) #ti values
    #loading spectra
    S_x_2[i] = S_data_2[i]['lambda'] #wavelength
    S_x_2[i] = S_x_2[i][780:1050] #removing invalid wavelengths
    S_y_2[i] = S_data_2[i]['halo'] #halo values #shape = (20,2000,15)
    S_y_2[i] = np.sum(S_y_2[i],axis=2) # summing nstark #shape = (20,2000)
    S_y_2[i] = S_y_2[i][irho_2,780:1050] #sorting spectra and removing values of invalid wavelengths 
    S_y_2[i] = np.array(S_y_2[i]).flatten() #merging data from all LOS
        
    #shot 3
    #sorting rho_pol values
    irho_3 = np.argsort(P_data_3[i].rhop_los)
    #loading profiles
    T_x_3[i] = P_data_3[i].rhop_los[irho_3] #rho_pol
    T_y_3[i] = np.interp(T_x_3[i][irho_3], P_data_3[i].profFIDASIM['rhop'], P_data_3[i].profFIDASIM['ti']) #ti values
    #loading spectra
    S_x_3[i] = S_data_3[i]['lambda'] #wavelength
    S_x_3[i] = S_x_3[i][780:1050] #removing invalid wavelengths
    S_y_3[i] = S_data_3[i]['halo'] #halo values #shape = (20,2000,15)
    S_y_3[i] = np.sum(S_y_3[i],axis=2) # summing nstark #shape = (20,2000)
    S_y_3[i] = S_y_3[i][irho_3,780:1050] #sorting spectra and removing values of invalid wavelengths 
    S_y_3[i] = np.array(S_y_3[i]).flatten() #merging data from all LOS


print('-------------')
print('Components of data loaded')
    
Lambda = S_x_1[0]
Rho = T_x_1[0]
LOS = T_x_1[0]

# %% - #Loading original spectra and profiles (in similar way) to be used for prediction 
#These are files that weren't scanned or labelled LScan1

no_ti_scans = 1
no_scans = 9
ti_scan = ['1.00']*no_scans #multiply by number of elements in scan_number #to load scan ti 0.25-5.00
scan_number = ['1','2', '3','4', '5','6','7','8','9']*no_ti_scans #multiply by number of elements in ti_scan #to load scans LScan1-9
number = no_ti_scans*no_scans #count number of scans to be loaded per shot
shots = ['38587'] #'35840','38539'
n_shots = len(shots)
total_number = n_shots*number
print('Total number of shots to be loaded is: ', total_number)
print(' ')
#Loading data
#for spectra
O_spec_1 = [ [] for _ in range(number) ] #for spectra of original profiles 
Empties = []
for i in range(number):
    try:
        #Loading spectra bin
        O_spec_1[i] = readS_1('BOS_38587_LScan%s_ti_%s'%(scan_number[i],ti_scan[i]))            
    except FileNotFoundError:
        print('A file was not found')
        Empties.append(i)
        print('Data point %s to be removed from loaded list'%(i))
        pass   
    
#for profiles
rootO = '/afs/ipp/home/d/debsel/FIDASIM4/RESULTS/Scans/Linear/38587/'
id_O = [ [] for _ in range(number) ] 
P_data_O = [ [] for _ in range(number) ] 

for i in range(number):
    try:
        #make id for each file
        #Profiles
        id_O[i] = 'BOS_38587_LScan%s_ti_%s'%(scan_number[i],ti_scan[i])        
        P_data_O[i] = SynSignal(rootO + id_O[i])
        
    except FileNotFoundError:
        print('A file was not found')
        Empties.append(i)
        print('Data point %s to be removed from loaded list'%(i))
        pass

New_number = len(O_spec_1)

#Selecting components
O_S_x_1 = [ [] for _ in range(New_number) ] #sets of lambda
O_S_y_1 = [ [] for _ in range(New_number) ] #sets of spectra
#Profile
O_T_x_1 = [ [] for _ in range(New_number) ] #sets of lambda
O_T_y_1 = [ [] for _ in range(New_number) ] #sets of spectra


for i in range(New_number):
    #shot 1
    #sorting rho_pol values
    irho_O = np.argsort(P_data_O[i].rhop_los)
    #loading profiles
    O_T_x_1[i] = P_data_O[i].rhop_los[irho_O] #rho_pol
    O_T_y_1[i] = np.interp(O_T_x_1[i][irho_O], P_data_O[i].profFIDASIM['rhop'], P_data_O[i].profFIDASIM['ti']) #ti values
    #loading spectra
    O_S_x_1[i] = O_spec_1[i]['lambda'] #wavelength
    O_S_x_1[i] = O_S_x_1[i][780:1050] #removing invalid wavelengths
    O_S_y_1[i] = O_spec_1[i]['halo'] #halo values #shape = (20,2000,15)
    O_S_y_1[i] = np.sum(O_S_y_1[i],axis=2) # summing nstark #shape = (20,2000)
    O_S_y_1[i] = O_S_y_1[i][irho_O,780:1050] #sorting spectra and removing values of invalid wavelengths 
    O_S_y_1[i] = np.array(O_S_y_1[i]).flatten() #merging data from all LOS   




# #Using Data from ALL LOS
Spectra_data_all = S_y_1 + S_y_2 + S_y_3
Temp_data_all = T_y_1 + T_y_2 + T_y_3 

n_shots = 3

print('Shape of spectral data: ', np.shape(Spectra_data_all) )
print('Shape of profile data: ', np.shape(Temp_data_all) )   


#Using test_train_split to divide dataset
X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(Spectra_data_all, Temp_data_all, train_size = 0.7, random_state = 1)

#Scaling Data
scaler = StandardScaler()
X_train_scaled_all = scaler.fit_transform(X_train_all)
X_test_scaled_all = scaler.fit_transform(X_test_all)
Y_train_scaled_all = scaler.fit_transform(Y_train_all)
Y_test_scaled_all = scaler.fit_transform(Y_test_all)
O_spec_scaled = scaler.fit_transform(O_S_y_1)
O_temp_scaled = scaler.fit_transform(O_T_y_1)

#Defining an MLP model
MLP_all = Sequential()
MLP_all.add(Dense(600, input_dim = (5400), activation = 'relu'))
MLP_all.add(Dense(20, activation = 'linear'))
opt = adam_v2.Adam(learning_rate = 0.001, decay = 1e-6) #defining optimizer parameters
MLP_all.compile(loss='mean_squared_error', optimizer=opt) #Compiling model #metrics=['mse']

#Fit model on dataset
Fit_all = MLP_all.fit(X_train_scaled_all, Y_train_scaled_all, validation_data = (X_test_scaled_all, Y_test_scaled_all), epochs=10, verbose = 1) #batch_size=2 #validation_split = 0.1
##epoch = n_iterations, batch_size = no of rows to be considered before updating weights
##validation_split reserves some data for checking generalization of model, useful to evaluate model

MLP_all.summary()

# #Evaluate Model
print('')
print('Evaluating model on training and test data')
print('------------------------------')
train_mse_all = MLP_all.evaluate(X_train_scaled_all, Y_train_scaled_all, verbose = 1)
test_mse_all = MLP_all.evaluate(X_test_scaled_all, Y_test_scaled_all, verbose = 1)
print('Train MSE: %.3f , Test MSE: %.3f '%(train_mse_all, test_mse_all)) #loss = mse*100

#Make predictions on test data
print('')
print('Making prediction on test data')
print('------------------------------')
MLP_predict_all = MLP_all.predict(O_spec_scaled) 
print('Model outputs a shape of: ', np.shape(MLP_predict_all))
print('Correct shape of output is: ',np.shape(O_temp_scaled)) 
  
#Checking for numerical output values from model
print('')
print('Confirming numerical output from model')
print('------------------------------')
print('Max. numerical value in output of model is: ',np.max(MLP_predict_all))
print('Min. numerical value in output of model is: ',np.min(MLP_predict_all))

#PREDICTION PLOT
Model_all = scaler.inverse_transform(MLP_predict_all)
Y_all= scaler.inverse_transform(O_temp_scaled) 
plt.figure(figsize = (9,7))
plt.plot(LOS, Y_all[4], 'go', label = 'Synthetic profile') 
plt.plot(LOS, Model_all[4], 'ro', label = 'Prediction (using flattened data from all LOS)') # '--'
plt.xlabel('\u03C1', fontsize = 18)
plt.ylabel('T{} [keV]'.format(get_sub('i')), fontsize =18)
plt.rc('xtick', labelsize= 17)    
plt.rc('ytick', labelsize= 17)
plt.title('Predicting T{}'.format(get_sub('i')), fontsize = 20)
plt.legend(fontsize = 17)
plt.show()



