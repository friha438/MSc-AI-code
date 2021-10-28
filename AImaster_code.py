import numpy as np
import pandas as pd
import math
import random
import bisect 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from ngboost import NGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.isotonic import IsotonicRegression

pd.options.mode.chained_assignment = None

#############################
#       Read data           #
#############################

#read and clean data
def read_data():

    # Load four sheets from excel file
    denver = pd.read_excel('Datas.xls', sheet_name='DEN')
    newyork = pd.read_excel('Datas.xls', sheet_name='NYC')
    sanfrancisco = pd.read_excel('Datas.xls', sheet_name='SFC')
    texas = pd.read_excel('Datas.xls', sheet_name='TXS')

    random.seed(48)

    # Cleen data 
    denver['station'] = denver['station'].replace(np.nan,'DENVER INTL AP')
    denver['station'] = denver['station'].replace('DENVER INTL AP', 'DEN')
    newyork['station'] = newyork['station'].replace(np.nan, 'JFK')
    sfc = sanfrancisco.drop(columns = 'valid', axis=1)

    return denver, newyork, sfc, texas

#Function to rename columns 
def rename_columns(df, col_names):
    cols = df.columns
    for col, new_col in zip(cols, col_names):
        df.rename(columns={col:new_col}, inplace=True)

#rename all columns
def rename_all(denver, newyork, sfc, texas):
    # Rename columns in all datasets to same column names
    col_names = ['station', 'skycloud', 'day', 'tmpf', 'feel', 
                'zenith', 'azimuth', 'glob', 'direct', 
                'diffused', 'albedo', 'time', 'power']

    rename_columns(denver, col_names)
    rename_columns(newyork, col_names)
    rename_columns(sfc, col_names)
    rename_columns(texas, col_names)

#create dataset with all stations
def concat_frames():
    # Join all 4 datasets into a single dataframe
    frames = [denver, newyork, sfc, texas]
    data = pd.concat(frames)

    # Create dataset for denver station
    den = data.loc[data['station'] == 'DEN']

    return den

#choose features
def choose_features(den):
    # Drop columns with low correlation
    den = den.drop(columns = ['skycloud', 'feel', 'albedo'], axis=1)

    # Drop rows with azimuth = -99 values
    den.drop(den.index[den['azimuth'] == -99], inplace = True)

    return den

denver, newyork, sfc, texas = read_data()
rename_all(denver, newyork, sfc, texas)
denv = concat_frames()
den = choose_features(denv)


#############################
#       Split data          #
#############################

#Split data for training and testing
def split_train_test_data(den):

    # Create X and y values
    X_val = den.drop(columns = ['station', 'power','day'], axis=1) 
    X = X_val.values
    y = den.power.values

    # Train test split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=300, 
        random_state=48, shuffle=True)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return scale_data(X_train, y_train, X_test, y_test)

#Scale train and test data
def scale_data(X_train, y_train, X_test, y_test):
    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    y_tr_sc = scaler.fit_transform(y_train)
    y_te_sc = scaler.transform(y_test)

    return X_tr_sc, X_te_sc, y_tr_sc, y_te_sc

#Split data for testing and calibration
def split_test_calib_data(X_tr_sc, X_te_sc, y_tr_sc, y_te_sc):
 
    #Split into test and calibration sets
    X_calib, X_true_test, y_calib, y_true_test = train_test_split(X_te_sc, y_te_sc, 
        test_size=0.5, random_state=48, shuffle=True)

    return X_calib, X_true_test, y_calib, y_true_test 

X_tr_sc, X_te_sc, y_tr_sc, y_te_sc = split_train_test_data(den)
X_calib, X_true_test, y_calib, y_true_test = split_test_calib_data(X_tr_sc, X_te_sc, y_tr_sc, y_te_sc )


##########################################
#       Create forecasting models        #
##########################################

#Set parameters for GPR
def set_parameters(X_tr_sc):

    # Create length scale and length scale bounds
    feature_len = len(X_tr_sc[0])
    len_scale = np.ones(feature_len)

    hp_low = 1e-10
    hp_high = 100000

    hp_bounds = np.zeros(shape = (feature_len,2))
    hp_bounds[:,0] = hp_low
    hp_bounds[:,1] = hp_high

    return len_scale, hp_bounds

# sklearn gpr model matern 52 kernel + white kernel
def gpr_model(len_sc, hp_b, X_tr, y_tr, X_cal, X_test):
    kernel = Matern(length_scale=len_sc, length_scale_bounds=hp_b, nu=2.5) + WhiteKernel(noise_level=0.1)
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X_tr, y_tr)
    mean_calib, std_calib = model.predict(X_cal, return_std=True)
    mean_true_test, std_true_test = model.predict(X_test, return_std=True)
    
    return mean_calib, std_calib, mean_true_test, std_true_test

#train and predict using the gpr model
len_scale, hp_bounds = set_parameters(X_tr_sc)
gpr_mean_calib, gpr_std_calib, gpr_mean_true_test, gpr_std_true_test = gpr_model(len_scale, hp_bounds, 
	X_tr_sc, y_tr_sc, X_calib, X_true_test)

#NgBoost model
def ngb_model(X_tr, y_tr, X_cal, X_test):
    ngb = NGBRegressor()
    ngb.fit(X_tr, y_tr)
    ngb_preds_calib = ngb.pred_dist(X_cal)
    ngb_preds_test = ngb.pred_dist(X_test)
    
    mean_calib = ngb_preds_calib.params['loc']
    std_calib = ngb_preds_calib.params['scale']
    mean_test = ngb_preds_test.params['loc']
    std_test = ngb_preds_test.params['scale']
    
    return mean_calib, std_calib, mean_test, std_test

#Train and predict using the ngboost model
ngb_mean_calib, ngb_std_calib, ngb_mean_true_test, ngb_std_true_test = ngb_model(X_tr_sc, y_tr_sc, X_calib, X_true_test)


###################################
#       CRUDE calibration         #
###################################

#Get z scores from the predicted data
def find_z_scores(y_calib, mean_calib, std_calib):
    z_calib = []

    for i in range(len(mean_calib)):
        dif = y_calib[i] - mean_calib[i] 
        z_st = dif/std_calib[i]
        bisect.insort(z_calib, z_st) #sorted list of z_star values

    # This is noted as Zc in the article
    z_c = np.concatenate(z_calib, axis=0 )  
    
    return z_c

# function to find quantiles, p is the target quantile level
def CRUDE(z_c, mean_calib):
    p_values = np.arange(0.05,1,0.05)
    
    # empty 2d array for z_p scores for all p values
    z_p = np.zeros((len(p_values), 2))
    
    for p in range(len(p_values)):
        
        #define higher and lower boundries
        p_up = round(p_values[p]+(1-p_values[p])/2, 3)
        p_low = round(1-p_up, 3)

        #determine which Zc values to include in both boundries
        z_p_up = z_c[int(p_up * len(mean_calib))]
        z_p_low = z_c[int(p_low * len(mean_calib))]
        
        z_p[p,0] = z_p_low
        z_p[p,1] = z_p_up
        
    return z_p

# Get quantiles based on z scores
def find_quantiles(z_p, mean_true_test, std_true_test):
 
    # empty 2*z_p array for lower and higher boundry values
    q = np.zeros((len(mean_true_test), 2*len(z_p)))
    index = 0
    
    for j in range(len(z_p)):  
        for i in range(len(mean_true_test)):
            #Define the lower and higher boundry for each data point
            q_low = mean_true_test[i] + std_true_test[i] * z_p[j,0]
            q_up = mean_true_test[i] + std_true_test[i] * z_p[j,1]
            
            q[i,index] = q_low
            q[i,index + 1] = q_up
        
        index = index + 2       
    return q

#Calculate the coverage
def crude_coverage(q, y_true_test):

    p_values = np.arange(0.05,1,0.05)
    cov = []
    index = 0

    for p in p_values:
        count = 0
        for i in range(len(q)):
            if (y_true_test[i] > q[i,index]) and (y_true_test[i] < q[i,index+1]):
                count = count + 1 
        index = index + 2
        cov.append(count/len(q))
        
    return cov

# CRUDE applied on GPR model
z_c_gpr = find_z_scores(y_calib, gpr_mean_calib, gpr_std_calib)
z_p_gpr = CRUDE(z_c_gpr, gpr_mean_calib)
q_gpr = find_quantiles(z_p_gpr, gpr_mean_true_test, gpr_std_true_test)
cov_crude_gpr = crude_coverage(q_gpr, y_true_test)

# CRUDE applied on NgBoost model
z_c_ngb = find_z_scores(y_calib, ngb_mean_calib, ngb_std_calib)
z_p_ngb = CRUDE(z_c_ngb, ngb_mean_calib)
q_ngb = find_quantiles(z_p_ngb, ngb_mean_true_test, ngb_std_true_test)
cov_crude_ngb = crude_coverage(q_ngb, y_true_test)


######################################
#       Kuleshov calibration         #
######################################

# Get the confidence intervals for the predictions
def find_conf_interval(mean_calib, y_calib, std_calib):

    cdfs = []

    for i in range(len(mean_calib)):
        dif = y_calib[i] - mean_calib[i] 
        z_st = dif/std_calib[i]
        cdf = stats.norm.cdf(z_st)
        cdfs.append(cdf)

         #if cdf < 0.5:
         #    centered.append(1 - 2*cdf)
         #else:
         #    centered.append(cdf - (1-cdf))

    # This is the Ft from the article
    CI_cent = np.concatenate(cdfs, axis=0 )   
    
    return CI_cent

#Get the coverage of the predictions
def find_coverage(CI_cent):
    
    cover_cent = []

    for i in range(len(CI_cent)):
        count = 0
        for j in range(len(CI_cent)):
            if CI_cent[j] < CI_cent[i]:
                count = count +1
        cov_cent = count/len(CI_cent)
        cover_cent.append(cov_cent)
    
    return cover_cent

# Do the Kuleshov calibration
def train_ir_calc_coverage(CI_cent, cover_cent, mean_true_test, y_true_test, std_true_test):

    ir = IsotonicRegression()
    # empirical (H) first and predicted (P) next
    h = ir.fit_transform(CI_cent, cover_cent)

    coverage_cal_cent = []
    coverage_uncal_cent = []

    p_values = np.arange(0.05,1,0.05)
    cdf_cals = np.zeros((len(mean_true_test), 1))
    cdf_uncals = np.zeros((len(mean_true_test), 1))
    limits = np.zeros((len(mean_true_test), len(p_values)))
    
    q_upper_init = 2.5*(max(mean_true_test) - min(mean_true_test))
    
    for p in reversed(range(len(p_values))):
        count_cal = 0
        count_cal_centered = 0
        count_uncal = 0
        
        #get upper and lower p values
        p_up = round(p_values[p]+(1-p_values[p])/2, 3)
        p_low = round(1-p_up, 3)
        
        for j in range(len(mean_true_test)):
            dif = y_true_test[j] - mean_true_test[j]
            z_st = dif/std_true_test[j]   
            cdf_uncal = stats.norm.cdf(z_st)
            
            tol = 0.001
            delta_mid = 2*tol
            q_lower = 0
            q_upper = q_upper_init
            if p < len(p_values) - 1:
                q_upper = limits[j, p+1]
            q_mid = 0.5 * (q_lower + q_upper)
            
            #recalibrate the cdfs using the trained isotonic regression model
            cdf_cal = ir.predict(cdf_uncal)
            
            while abs(delta_mid) > tol: 
                q_mid = 0.5 * (q_lower + q_upper)
                
                e_low = stats.norm.cdf(-q_mid/std_true_test[j])
                e_upp = stats.norm.cdf(q_mid/std_true_test[j])
                
                e_low = e_low.reshape((1, 1))
                e_upp = e_upp.reshape((1, 1))
                
                s_low = ir.predict(e_low)
                if np.isnan(s_low):
                    s_low = 0
                    
                s_upp = ir.predict(e_upp)
                if np.isnan(s_upp):
                    s_upp = 1
                
                true_ci = s_upp - s_low
                delta_mid = true_ci - p_values[p]

                if delta_mid > 0:
                    q_upper = q_mid
                else:
                    q_lower = q_mid
                    
            limits[j, p] = q_mid
            
            #Get cdfs, both calculated and uncalculated
            cdf_cals[j,0] = cdf_cal
            cdf_uncals[j,0] = cdf_uncal

            #calculate coverage
            if cdf_cal <= p_values[p]:
                count_cal = count_cal+1
                
            if y_true_test[j] < mean_true_test[j] + limits[j,p] and y_true_test[j] > mean_true_test[j] - limits[j,p]:
                count_cal_centered = count_cal_centered+1 

            if cdf_uncal <= p_values[p]:
                count_uncal = count_uncal+1


        coverage_cal_cent.append(count_cal/len(mean_true_test))
        coverage_uncal_cent.append(count_uncal/len(mean_true_test))
        print('coverages calculated')
        
    return coverage_cal_cent, coverage_uncal_cent, cdf_cals, cdf_uncals, limits

# Get Kuleshov recalibrated intervals for NgBoost
conf_int_ngb = find_conf_interval(ngb_mean_calib, y_calib, ngb_std_calib)
coverage_ngb = find_coverage(conf_int_ngb)

cov_kuleshov_calc_ngb, cov_uncal_ngb, cdf_cals_ngb, cdf_uncals_ngb, limits_ngb = train_ir_calc_coverage(conf_int_ngb, coverage_ngb, 
	ngb_mean_true_test, y_true_test, ngb_std_true_test)

# Get Kuleshov recalibrated intervals for GPR
conf_int_gpr = find_conf_interval(gpr_mean_calib, y_calib, gpr_std_calib)
coverage_gpr = find_coverage(conf_int_gpr)

cov_kuleshov_calc_gpr, cov_uncal_gpr, cdf_cals_gpr, cdf_uncals_gpr, limits_gpr = train_ir_calc_coverage(conf_int_gpr, coverage_gpr, 
 	gpr_mean_true_test, y_true_test, gpr_std_true_test)


##################################################
#       Improve sharpness of calibrations        #
##################################################

# Save upper and lower limits for both CRUDE and Kuleshov in the same format
def ranges(q, mean, limits):
    kuleshov_range = np.zeros((len(q), len(q[0])))

    index = 0
    for p in range(len(limits[0])):
        for i in range(len(q)):
            kuleshov_range[i,index] = mean[i] - limits[i,p]
            kuleshov_range[i,index+1] = mean[i] + limits[i,p]
        index = index + 2
    
    return q, kuleshov_range

ngb_crude_range, ngb_kuleshov_range = ranges(q_ngb, ngb_mean_true_test, limits_ngb)
gpr_crude_range, gpr_kuleshov_range = ranges(q_gpr, gpr_mean_true_test, limits_gpr)


#Function to create weights and apply them on ranges
def get_weighted_ranges(crude_range, kuleshov_range, conf_limit):

    # Set two weights, lower and upper, each ranging from 0 to 1 changing 0.1 at a time
    w_lower = np.arange(0.0,1.1,0.1)
    w_upper = np.arange(0.0,1.1,0.1)

    # Save the final values in one array for upper + lower boundries, and one for the interval
    w_ranges = np.zeros((len(crude_range), 2*len(w_upper)*len(w_lower)))
    w_interval = np.zeros((len(crude_range), len(w_upper)*len(w_lower)))

    index1 = 0
    index2 = 0
    
    # Iterate through each combination of w_upper and w_lower
    for w1 in w_lower:
        for w2 in w_upper: 
            # Iterate through all values in the calibrated ranges data sets
            for i in range(len(crude_range)):
                # Calculate an upper and lower boundry based on previous ranges and weights
                lb = w1 * kuleshov_range[i,conf_limit] + (1-w1) * crude_range[i,conf_limit]
                ub = w2 * kuleshov_range[i,conf_limit +1] + (1-w2) * crude_range[i,conf_limit +1]

           #      if i == 0:
           #          print(w1, w2, kuleshov_range[i,conf_limit], kuleshov_range[i,conf_limit+1], 
           #                crude_range[i,conf_limit], crude_range[i,conf_limit +1])
           #          print(lb, ub)
                # Save resulting interval and lower + upper boundry for each point
                w_interval[i, index1] = ub - lb
                w_ranges[i, index2] = lb
                w_ranges[i, index2+1] = ub

            index1 = index1 + 1
            index2 = index2 + 2

    # Returns an array 2249 values long and each point contains 121 intervals/ 242 ranges
    return w_interval, w_ranges

#Apply weighted ranges (so far only on NgBoost)
def apply_weights(ngb_crude_range, ngb_kuleshov_range):
    c_lim = np.arange(0,len(ngb_crude_range[0]),2)
    weight_intervals_ngb = []
    weight_ranges_ngb = []

    # Iterate through all confidence limits (0.05 to 0.95)
    for i in c_lim:
        w_interval_ngb, w_ranges_ngb = get_weighted_ranges(ngb_crude_range, ngb_kuleshov_range, i)
        
        # Save results in a list of intervals/ranges
        weight_intervals_ngb.append(w_interval_ngb)
        weight_ranges_ngb.append(w_ranges_ngb)

    return weight_ranges_ngb, weight_intervals_ngb

#Function to calculate coverage using the weighted ranges    
def get_coverage_weighted(w_interval, w_ranges, y_calib, mean_calib):

    index = 0
    counter = []
    
    # Iterate through each weight combination
    for w in range(len(w_interval[0])):
        count = 0
        # Iterate through each point (2249 points)
        for i in range(len(w_ranges)):

            # Check if the prediction is in the interval
            if y_calib[i] > w_ranges[i, index] and y_calib[i] < w_ranges[i, index + 1]:
                count = count + 1
                            
        # Divide by number of points to get coverage, save in counter for each weight combination
        counter.append(float(count)/float(len(w_ranges)))
        index = index + 2
    
    # Returns a list of all weight combination's coverage
    return counter

#Save coverage (of NgBoost)
def save_coverages(y_calib, ngb_mean_calib, weight_ranges_ngb, weight_intervals_ngb):
    coverages = []

    # Iterate through each confidence limit (0.05 to 0.95)
    for (ran, inter) in zip(weight_ranges_ngb, weight_intervals_ngb):

        # Get the coverage of each weight combination
        counter_ngb = get_coverage_weighted(inter, ran, y_calib, ngb_mean_calib)
        coverages.append(counter_ngb)

    return coverages

#Calculate coverage scores
def get_coverage_score(coverages):
    RMSE = np.zeros(len(coverages[0]))
    p_val = np.arange(0.05,1,0.05)

    for w in range(len(coverages[0])):
        for (p,c) in  zip(p_val,range(len(coverages))):
            cov = coverages[c]
            dif = cov[w] - p
                    
            RMSE[w] = RMSE[w] + np.sqrt(dif*dif)

    return RMSE

#Function to calculate sharpness
def calc_sharpness_weighted(w_interval):

    index = 0
    sharpness = []
    
    # Iterate through each weight combination
    for w in range(len(w_interval[0])):
        sum_ranges = 0
        
        # Iterate through each point (2249 points)
        for i in range(len(w_interval)):   
            sum_ranges = sum_ranges + w_interval[i,w] 

        #print(sum_ranges, sum_ranges/len(w_interval))
        sharpness.append(sum_ranges/len(w_interval))
    
    # Returns a list of all weight combination's sharpness
    return sharpness

#Save all sharpness scores
def get_sharpness_scores(weight_intervals_ngb):

    sharpness = []
    for i in range(len(weight_intervals_ngb)):
        sharpness.append(calc_sharpness_weighted(weight_intervals_ngb[i]))
        
    sharpness = np.array(sharpness)


    sharp_score = []
    for w in range(len(sharpness[0])):
        sum_sharp = 0
        for p in range(len(sharpness)-1):
            sum_sharp = sum_sharp + sharpness[p,w]
            
        #print((sharpness[0,w] + sharpness[1,w] + sharpness[2,w] + sharpness[3,w])/4)
        sharp_score.append(sum_sharp/(len(sharpness)-1))

    return sharp_score

weight_ranges_ngb, weight_intervals_ngb = apply_weights(ngb_crude_range, ngb_kuleshov_range)
coverages = save_coverages(y_true_test, ngb_mean_true_test, weight_ranges_ngb, weight_intervals_ngb)
cov_score = get_coverage_score(coverages)
sharp_score = get_sharpness_scores(weight_intervals_ngb)


fig, ax = plt.subplots(1, 1, figsize=(12, 5))
plt.plot(sharp_score, cov_score, 'o')
plt.xlabel('Sharpness score')
plt.ylabel('Calibration score')

plt.show()
