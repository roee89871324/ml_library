"""
Summary: Machine learning library. It is a bit messy at times because only intended for personal use. 
    Don't shy from continueing  to update this library 'on the fly'.

contains:
1. own implementation of PCA
2. own implementation of GMM model
3. own implementation of k-means model
4. efficient distance and other metric calculations
5. normalization 
6. evaluation / objective functions (softmax, sigmoid, hitrate, )
7. functions to simplify hyper-parameter tuning
8. feature importance (lasso, co-correlation, decision trees)
9. outlier removal functions
10. feature augmentation (one hot encoding, binning features)
11. Cleaning/fixing data (missing value filling, etc)
12. many evaluation functions 
"""

from __future__ import print_function
from time import time
import os
import numpy
import pandas
from scipy.stats.stats import pearsonr
import random
import torch
import numpy
from time import time 
import math
from load_my_data_set import *
from disp_one import *
from montage import montage
import scipy.io
import matplotlib.pyplot as plt
from logdet import logdet
import warnings
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns
# import pandas, numpy, sys
# from sklearn.model_selection import cross_val_score
# from scipy.stats.stats import pearsonr
# import sklearn
# import seaborn as sns
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from collections import Counter
# import copy, six
# # import scipy.stats
# from itertools import chain, combinations
# from sklearn import linear_model
# import random
# # import scipy.io
# import pickle
# from kaggalgo import *

# consts
PERCENTILE = 1
FUNC = 0
OBS_NUM = 0
COL_NUM = 1
ROWS = 1
ONE_DIM = 1
TWO_DIM = 2
BEST_CONFIG_POS = 0
ADD_COLUMN = 1
EXTEND_ROW = 0
INTEGER = 0
FLOAT = 0.1
NUM_ROWS = 0
FRACTION_TO_PERCENTAGE = 100
TWO_DIMS = 2
ONE_DIM = 1

####### REINFORCEMENT LEARNING #######
def normalize_vec(v_1d):
    return (v_1d - v_1d.mean()) / v_1d.std()

def save_weights(agent, envName, timestamp):
    filename = r"C:/home/GitHub/honorsProject/modelWeights/" + envName + ("_%d.torch" % (timestamp, ))
    torch.save(agent.state_dict(), filename )

def envinfo(env):
    print (env.observation_space)
    print (env.action_space)
    print (dir(env.spec))
    print (env.spec.reward_threshold)

def RL_seed(env=None, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if (env): env.seed(seed)

####### NEURALNETS #######
def bin_classif_cost(y_est_1d, y_true_1d):
    return -1 * ( numpy.log(y_est_1d)*y_true_1d + numpy.log(1-y_est_1d)*(1-y_true_1d) ).mean()

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = x_exp.sum(axis=1, keepdims=True)
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    
    return s

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    # Divide x by its norm.
    x = x / x_norm

    return x

def image2vector(image_2d):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image_2d.reshape(image_2d.shape[0], -1)
    
    return v

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    s = 1.0/(1+np.exp(-1 * x))
    
    return s

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    s = sigmoid(x)
    ds = s * (1 - s)
    
    return ds

####### DATA #######
# make ests matrix
def _test_autocorrelation(starting_autocorr, shuffled_y_1d):
    # summary: test autocorrelation to know gorupping was done well.

    if abs(starting_autocorr) > 0.2 and (abs(autocorr(shuffled_y_1d)) < 0.05): 
        warnings.warn("autocorrelation changed in shuffle.")

def _shuffle_group_rows(x_2d, y_1d, groupping_1d):
    # summary: shuffle rows in groups according to group id (rows with same groups id remain close to eachother
    # param obs_grouping_1d: the id of the group every observation belongs to (so it would be shuffled together)

    if (len(x_2d) != len(y_1d)) or ((groupping_1d is not None) and (len(y_1d) != len(groupping_1d))):
        raise Exception("Groupping_1d or y_1d or x_2d do not have same num of rows.")

    # calculate the auto correlation at the beggining for testing purposes
    starting_autocorr = autocorr(y_1d)

    # add label to the x_2d
    x_2d = pandas.DataFrame(x_2d).assign(y_1d=pandas.Series(y_1d))

    # shuffle by groups
    if groupping_1d is not None:
        x_2d = x_2d.assign(group_ID=groupping_1d)
        grouped_rows_list = [df for _, df in x_2d.groupby('group_ID')]
        random.shuffle(grouped_rows_list)
        shuffled_x_2d = pandas.concat(grouped_rows_list)
        shuffled_x_2d.pop("group_ID")
    else:
        shuffled_x_2d = x_2d.sample(frac=1)

    # get indices
    shuffled_indices_1d = numpy.array(shuffled_x_2d.index)
    shuffled_x_2d = shuffled_x_2d.reset_index(drop=True)
    
    # fix shuffled matrix
    shuffled_y_1d = shuffled_x_2d.pop("y_1d")
    shuffled_x_2d = numpy.array(shuffled_x_2d)
    
    _test_autocorrelation(starting_autocorr, shuffled_y_1d)

    return shuffled_x_2d, shuffled_y_1d, shuffled_indices_1d

def _get_indices(fold_start_pos, fold_end_pos, num_of_obs):
    # summary: return indices of current fold and the rest in the data. ([0, 1, 2, 6, 7, 8, 9], [3, 4, 5])->([0, 1, 2, 6, 7, 8, 9], [3, 4, 5])

    # if pred larger than total num of obs, then just take till the end (num of obs)
    if fold_end_pos > num_of_obs: fold_end_pos = num_of_obs
    fold_indices = range(fold_start_pos, fold_end_pos)
    
    # get leftover indices
    all_indices = range(0, num_of_obs)
    leftover_idices = list( (set(all_indices) - set(fold_indices)) )

    return leftover_idices, fold_indices

def _split_data_to_folds(num_folds, x_train_2d, y_train_1d):
    # summary: splits data to a list of data sections per fold (x_train_2d, y_train_1d, x_pred_2d, y_pred_2d)

    if (num_folds <= 1) or (num_folds > len(x_train_2d)) or (len(x_train_2d) != len(y_train_1d)): raise Exception("Invalid params for split data to folds! (roee exception :) )")

    fold_size, fold_start_pos, num_of_obs = len(x_train_2d) / float(num_folds), 0, len(x_train_2d)
    folds_data, x_train_2d, y_train_1d = [], numpy.array(x_train_2d), numpy.array(y_train_1d)
    # create folds data. minus epsilon is because sometime it's like .9999999999 and it still goes in unjustly.
    while float(str(fold_start_pos)) < num_of_obs:
        # define indices of x rows of train/pred. If last slice then just take everything that's left.
        train_row_indices, pred_row_indices = _get_indices(int(float(str(fold_start_pos))), int(float(str(fold_start_pos + fold_size))), num_of_obs)
        # folds_data.append((self.x_train_2d.iloc[train_row_indices].as_matrix(), self.y_train_1d.iloc[train_row_indices].as_matrix(), self.x_train_2d.iloc[pred_row_indices].as_matrix(), self.y_train_1d.iloc[pred_row_indices].as_matrix()))
        fold_x_train_2d, fold_y_train_1d, fold_x_pred_2d, fold_y_pred_1d = x_train_2d[train_row_indices], y_train_1d[train_row_indices], x_train_2d[pred_row_indices], y_train_1d[pred_row_indices]
        folds_data.append((fold_x_train_2d, fold_y_train_1d, fold_x_pred_2d, fold_y_pred_1d))
        # update fold starting index
        fold_start_pos += fold_size

        # test folds
        train_rows_num, pred_rows_num = fold_x_train_2d.shape[NUM_ROWS], fold_x_pred_2d.shape[NUM_ROWS]                
        if (pred_rows_num == 0) or (pred_rows_num < fold_size - 2) or (pred_rows_num > fold_size + 2) or (pred_rows_num > train_rows_num + 2) or (train_rows_num == 0): raise Exception("At least one folds went bad (impossible number of rows)")

    if len(folds_data) != num_folds: raise Exception("Too many/little folds! %s!=%s" % (len(folds_data), num_folds))

    return folds_data

# MODEL SELECTION
def eval_model(regr, eval_func, seed, build_train_params={}, shuffle_iter=1, groupping_1d=None, do_print=True):
    # summary: evaluate model in CV over several iterations
    # param iter: number of iterations to shuffle data and do kfold
    # param groupping_1d: vector by which groupping is made (rows with equal value are stick together after shuffle). If none then just shuffle normally.
    # example: {"eval_func":our_eval, "seed":random_state, "build_train_params": {"num_folds":2, "build_params":{"seed_iterations":10}}, "shuffle_iter":10, "groupping_1d":groupping_1d, "do_print":True}
    # example: eval_model(regr, ppt, params["seed"], build_train_params={"num_folds":2, "build_params":{"seed_iterations":10}}, shuffle_iter=10)

    set_seed_all(seed=seed, regr=regr)
    
    original_x_train_2d, original_y_train_1d = regr.x_train_2d, regr.y_train_1d
    folds_scores, prev_y_train_est_1d = [], numpy.array([])
    for i in range(shuffle_iter):
        # shuffled
        shuffled_x_train_2d, shuffled_y_train_1d, _ =  _shuffle_group_rows(original_x_train_2d, original_y_train_1d, groupping_1d=groupping_1d)
        # set
        regr.set_data(shuffled_x_train_2d, shuffled_y_train_1d)
        # build
        y_train_est_1d, fold_estimations = regr.build_train(**build_train_params)
        # print numpy.unique(y_train_est_1d)
        if len(y_train_est_1d) != len(shuffled_y_train_1d): raise Exception("Should be same length.!!!!! fix bug now. no delays.")
            
        # score
        folds_scores += map(lambda single_estimations: eval_func(*single_estimations), fold_estimations)

        if (groupping_1d is not None) and (len(numpy.unique(groupping_1d)) > 5) and numpy.array_equal(y_train_est_1d, prev_y_train_est_1d): raise Exception("Shuffle iterations wasted.")
        prev_y_train_est_1d = y_train_est_1d

    # return regr to what it was previously
    regr.set_data(original_x_train_2d, original_y_train_1d)

    # print formatted evaluation and return raw values.
    if do_print: print(analyze_vec(numpy.array(folds_scores)))
    # mean_val, min_val, max_val, std_val = analyze_vec(numpy.array(folds_scores), do_string=False)
    # return mean_val, min_val, max_val, std_val

def tune_param(regr, name_1, values_1, name_2="", values_2=[None], eval_params={}):
    ### summary: iterate over values of hyperparameter and print score. tune_param->eval_model->build_train->build
    # example: tune_param(regr, "eta", [1e-1,1e-2], eval_params={'eval_func':bin_classif_hitrate, 'seed':params["seed"], 'build_train_params':{'num_folds': 10}})

    # prevent changing existing regr inside function
    original_1 = regr.get_param(name_1)
    original_2 = regr.get_param(name_2) if name_2 != "" else None

    # iterate first hyperparameter
    for value_1 in values_1:
        print ("%s: %s" % (name_1, value_1))
        regr.set_param(name_1, value_1)

        # iterate second hyperparamter (if exists)
        for value_2 in values_2:
            # if hyperparameter exists.
            if value_2 != None: 
                print ("\t%s: %-5s <==> " % (name_2, value_2),)
                regr.set_param(name_2, value_2)

            # eval
            eval_model(regr, **eval_params)

    # set the parameters beack to what they were in the past
    regr.set_param(name_1, original_1)
    if original_2 is not None: regr.set_param(name_2, original_2)

    if name_1 == "eta" or name_2 == "eta": print ("REMEMBER TO UPDATE NROUNDS!!!!!!!!!!")

# feature selection
def fts_importance(regr, col_names, fts_num=None, do_plot=True, do_print=True):
    # summary: plot feature coefficients in the linear model. Note: model must be fitted to data

    if type(col_names[0]) != type(""): raise Exception("Only supports string col names.")

    fts_importance_1d = regr._get_fts_importance(col_names)
    # sort & get top vlues
    top_ft_importance_1d = kagglib.sort_series(fts_importance_1d, sort_abs=True)
    
    # show
    if fts_num: top_ft_importance_1d = top_ft_importance_1d.head(fts_num)
    if do_plot: 
        top_ft_importance_1d.plot(kind = "barh"); plt.title("Feature Coefficients"); plt.show()
    if do_print: 
        with pandas.option_context('display.max_rows', None, 'display.max_columns', 3):
            print (top_ft_importance_1d)

    return top_ft_importance_1d

def lasso_fts_importance():
    regr = sklearn_wrapper(linear_model.Lasso(alpha=1e-2), x_train_2d, y_train_1d, random_seed=random_state)  
    regr.fts_importance(fts_num=300, do_plot=False, do_print=False)

def corr_fts_importance(x_train_2d, y_train_1d, do_print=False):
    # summary: rating of the feature by correlation to label (in addition to other methods like rating by XGB or rating by LASSO or mutual information, etc.)
    
    # calc 
    fts_to_correlation = {}
    for ft_name in x_train_2d.columns:
        fts_to_correlation[ft_name] = abs(corr(x_train_2d[ft_name].values, y_train_1d.values.reshape(-1,)))

    # sort
    sorted_fts = sort_dict(fts_to_correlation, do_print=do_print)

    return sorted_fts, fts_to_correlation

# observation selection (outliers removal)
def rem_outliers_by_index(x_train_2d, y_train_1d, outliers=[]):
    # summary: delete a list of outliers (rows) from x and y
    # example: x_train_2d, y_train_1d = rem_outliers(x_train_2d, y_train_1d, [])

    # remove outliers from x & y by index
    pre_num_rows = len(x_train_2d)
    x_train_2d, y_train_1d = x_train_2d.drop(outliers), y_train_1d.drop(outliers)
    x_train_2d, y_train_1d = x_train_2d.reset_index(drop=True), y_train_1d.reset_index(drop=True)

    if len(x_train_2d) + len(outliers) != pre_num_rows: raise

    # check removal sucessful & return
    return x_train_2d, y_train_1d

def REWRITE_OUTLIERS_SECTION():
    def find_outliers_err_extremes(h_1d, drop_quantile=5, do_plot=False, do_print=False):
        # summary: find outliers by finding observations hardest to predict, assuming hard observations also confused model. 
        #     used mostly to remove extremes in error (between estimations and actual label) and extremes in y_train_1d
        # param drop_quantile: What percentage of outliers to "drop"
        # example: x_2d, y_train_1d = rem_outliers_by_index(x_2d, y_train_1d, outliers=find_outliers_extremes_1d(y_train_1d, drop_quantile=38))

        # sort
        h_1d = abs(h_1d)
        sorted_h_1d = pd.Series(h_1d).sort_values()

        # plot
        if do_plot: plt.scatter(pd.Series(range(len(sorted_h_1d))), sorted_h_1d); plt.show() 
        
        # get outliers
        num_of_outliers = int(len(sorted_h_1d) / 100.0 * drop_quantile)
        outliers_indices = list(sorted_h_1d.tail(num_of_outliers).index)

        # print method to remove the fuckers
        if do_print: print ("x_train_2d, y_train_1d = rem_outliers_index(x_train_2d, y_train_1d, %s)" % (outliers_indices, ))

        return outliers_indices

    def rem_outliers_err_extremes(regr, x_train_2d, y_train_1d, drop_quantile=5):
        # summary: remove the top percentage of observations to which algorithm had most error. Take some time to run so recommended to only use in the tunning phase to tune percentage we want to remove, then remove that percentage.
        # example: x_train_2d, y_train_1d = rem_outliers_err_extremes(my_xgb, x_train_2d, y_train_1d, drop_quantile=1)
        print ("i changed it abit check it still works")
        print ("isn't this redundant with rem_outliers_by_index and find_outliers_Extreme??")
        outliers_indices = find_outliers_extremes_1d(abs(regr.build_train(folds=2) - y_train_1d), drop_quantile=drop_quantile, do_plot=False)
        # remove 
        x_train_2d, y_train_1d = numpy.delete(x_train_2d, outliers_indices, axis=0), numpy.delete(y_train_1d, outliers_indices, axis=0)

        return x_train_2d, y_train_1d

    def find_outliers_X_graph(x_train_2d, y_train_1d, fts_num=10):
        # summary: prints graphs to help find outliers. click on dot to have the index printed to screen.
        # example: find_outliers(x_train_2d, y_train_1d, fts_num=30)

        print ("does this function ever bring good results? otherwise juts move it to deprecates")
        # print lowest 10 and highest 10 values to find outliers in label
        sorted_y_train_1d = y_train_1d.sort_values(); print ("%s %s" % (sorted_y_train_1d[:10], sorted_y_train_1d[-10:]))

        # get top features most correlated to Y
        y_fts_corr = {}
        for ft in x_train_2d.dtypes[x_train_2d.dtypes != "object"].index: y_fts_corr[ft] = abs(x_train_2d[ft].corr(y_train_1d))
        top_correlated_fts = dict(sorted(y_fts_corr.iteritems(), key=lambda x: x[1], reverse=True)[:fts_num]).keys()
        
        # scatterplot feature to label. set title & click on dot prints index
        outliers = []
        for ft in top_correlated_fts:
            outliers += plot_col_on_col(y_train_1d, x_1d=x_train_2d[ft], title=ft)

        # print collected outliers
        print ("x_2d, y_train_1d = rem_outliers(x_2d, y_train_1d, %s)" % (str(list(set(outliers))), ))

    def rem_outliers_X_extremes(x_2d, y_train_1d, top_quantile=0.999, ft_occurences=1):
        # summary: remove top percentage of outliers - of every feature. So if it removes 2 per feature and 200 features then 400.
        # param top_quantile: for every feature, count something as a potetial outlier only if it is in the top X quantile of values.
        # param ft_occurences: of the entire list of potential outliers, only remove it if it occured in at least x features
        # example: x_2d, y_train_1d = rem_outliers_x_extremes(x_2d, y_train_1d, top_quantile=.99, ft_occurences=10)
        # example: x_2d, y_train_1d = rem_outliers_x_extremes(x_2d, y_train_1d, top_quantile=.999, ft_occurences=2)

        print ("consider removing outliers by STD (like >std *6)")
        print ("does this function ever bring good results? otherwise juts move it to deprecates")
        print ("maybe us eth function find_outliers_extremes_1d")
        print ("does it use absolute value??")
        print ("consider combinging it with the find_outliers_x_graph and ingeenrally move it to deprecated i dont use it that much i think")
        # look for outliers only in train
        x_train_2d, x_pred_2d, potential_outliers = x_2d[:len(y_train_1d)], x_2d[len(y_train_1d):], []

        for ft in x_train_2d.columns:
            # find outliers indexes
            top_saf = x_train_2d[ft].quantile(top_quantile)
            ft_outliers_1d = x_train_2d[x_train_2d[ft] > top_saf].index
            # add
            potential_outliers += list(ft_outliers_1d)

        # keep only those that were potential outliers for several features
        outliers = [k for k, v in Counter(potential_outliers).iteritems() if v >= ft_occurences]
        # print len(outliers)
        # rem outliers
        x_train_2d, y_train_1d = rem_outliers_by_index(x_train_2d, y_train_1d, outliers=outliers)
        x_2d = pd.concat((x_train_2d, x_pred_2d)); x_2d = x_2d.reset_index(drop=True)

        return x_2d, y_train_1d

# normalize
def outliers_truncation(v_1d2d, std_factor=2.4):
    ## summary: truncate values above std*factor to that std*factor value

    if len(v_1d2d.shape) == 1: 
        truncated_v_1d = v_1d2d.copy()
        
        threshold = truncated_v_1d.std() * std_factor
        mean = truncated_v_1d.mean()
        
        truncated_v_1d[truncated_v_1d > mean + threshold] = mean + threshold
        truncated_v_1d[truncated_v_1d < mean - threshold] = mean - threshold

        return truncated_v_1d
    elif len(v_1d2d.shape) == 2: 
        # truncate each feature independently.
        
        for f in range(v_1d2d.shape[1]):
            v_1d2d[:, f] = outliers_truncation(v_1d2d[:, f], std_factor=std_factor)

        return v_1d2d
    else:
        raise

def _normalize_vector(v_1d, std_trunc, mean, div_by_std):
    v_1d = v_1d.copy()
    if std_trunc is not None:
        v_1d = outliers_truncation(v_1d, std_factor=std_trunc)
    if mean:
        v_1d = v_1d - v_1d.mean()    
    if div_by_std and (v_1d.std() != 0):
        v_1d = v_1d / v_1d.std()

    return v_1d

def normalize(v_1d2d, std_trunc=None, mean=True, div_by_std=True):
    # summary: normalize. Whether to truncate by std, reduce mean or divide by std.
    
    # if (len(v_1d2d.shape) == TWO_DIM) and (v_1d2d.shape[1] > v_1d2d.shape[0]):
    #     raise Exception("More features (%s) than observations (%s)." % (v_1d2d.shape[1], v_1d2d.shape[0]))
    if v_1d2d.dtype != numpy.float64 and v_1d2d.dtype != numpy.float32:
        raise Exception("Bad dtype:    %s" % (v_1d2d.dtype))
    if len(v_1d2d.shape) == ONE_DIM:
        raise Exception("Do not normalize Y like this - it deduct mean and divdes by std. fucks things up.")
    
    v_1d2d = v_1d2d.copy()

    if len(v_1d2d.shape) == ONE_DIM:
        return _normalize_vector(v_1d2d, std_trunc, mean, div_by_std)
    elif len(v_1d2d.shape) == TWO_DIM:
        for ft_index in range(v_1d2d.shape[1]):    
            v_1d2d[:, ft_index] = _normalize_vector(v_1d2d[:, ft_index], std_trunc, mean, div_by_std)
        return v_1d2d
    else:
        raise Exception("Support only 1 dim and 2 dim")

def onehot(v_1d):
    # summary: receive a 1d vector and convert it to 2d onehot encoded vector

    # assume number of possible values to be 0...max(v_1d) so if 3 is max, then 0,1,2,3 are possible.
    onehot_2d = numpy.zeros((len(v_1d), int(max(v_1d) + 1))) 
    onehot_2d[np.arange(len(v_1d)), v_1d] = 1

    return onehot_2d

def bin_ft(v_1d, percentile):
    binned_v_1d = numpy.zeros(len(v_1d))
    binned_v_1d[v_1d < (numpy.percentile(v_1d, percentile))] = -1
    binned_v_1d[v_1d > (numpy.percentile(v_1d, 100-percentile))] = 1
    
    return binned_v_1d

# missing
def test_is_reasonable_success(y_est_1d, y_true_1d):
    print ("num_zeros, len", numpy.count_nonzero(y_true_1d == 0), len(y_true_1d))
    print ("ppt_pos", ppt(y_est_1d[y_est_1d>0], y_true_1d[y_est_1d>0]))
    print ("ppt_neg", ppt(y_est_1d[y_est_1d<0], y_true_1d[y_est_1d<0]))
    print ("pnl, just_buying_pnl, pnl_if_100_hitrate, ratio_buying_perfect", pnl(y_est_1d, y_true_1d), pnl(numpy.ones(len(y_true_1d)), y_true_1d), pnl(numpy.sign(y_true_1d), y_true_1d), pnl(numpy.ones(len(y_true_1d)), y_true_1d) / pnl(numpy.sign(y_true_1d), y_true_1d))
    print ("max, min", max(y_true_1d), min(y_true_1d))
    print ("abs_mean, mean", numpy.mean(numpy.abs(y_true_1d)), numpy.mean(y_true_1d))
    print ("positives, negatives", numpy.count_nonzero(y_est_1d < 0), numpy.count_nonzero(y_est_1d > 0))
    # graph_cumsum(numpy.ones(len(y_true_1d)), y_true_1d)
    print

def nan_count(x_2d): 
    # summary: counts nans in every feature

    # calc total and percentage of Na in every feature
    bool_mask_2d = x_2d.isnull() 
    total_1d = bool_mask_2d.sum().sort_values(ascending=False); total_1d = total_1d[total_1d != 0]; percent_1d = total_1d / len(x_2d)
    
    return pd.concat([total_1d, percent_1d], axis=1, keys=['Total', 'Percent'])    

def nan_inplace_fill(x_2d, fts, val):
    # summary: fills nan values inplace according to give nvalue.
    # example: nan_inplace_fill(x_2d, ['BsmtFullBath', 'BsmtHalfBath'], 0)

    if val == "median":
        if (x_2d[fts].dtypes == "object").any(): raise Exception("Attempted to calc mean/median of numeric feature.")
        x_2d[fts] = x_2d[fts].fillna(value=x_2d[fts].median())

    elif val == "mean":
        if (x_2d[fts].dtypes == "object").any(): raise Exception("Attempted to calc mean/median of numeric feature.")
        x_2d[fts] = x_2d[fts].fillna(value=x_2d[fts].mean())

    else:
        x_2d[fts] = x_2d[fts].fillna(value=val)

####### EVAL #######
def analyze_vec(v_1d):
    v_1d = numpy.array(v_1d)
    if (numpy.count_nonzero(v_1d==0) / len(v_1d)) > .2: raise Exception("Too many zeros, did you do safs? This disregards this fact and would should very low mean ppt")
    return "%.3f (%.3f-%.3f, %.3f, %d)" % (numpy.mean(v_1d), numpy.min(v_1d), numpy.max(v_1d), numpy.std(v_1d), len(v_1d))

def eval_per_month(y_est_1d, y_true_1d, eval_func, n=20):
    # Summary: Split to n parts and see how many of those parts are negative...

    month_size = int(len(y_est_1d) / float(n))
    months = []
    for p in range(0, month_size*n, month_size):
        months.append(eval_func(y_est_1d[p:p+month_size], y_true_1d[p:p+month_size]))

    return numpy.array(months)

def pnl(y_est_1d2d, y_true_1d2d, profit_1d2d=None):
    # Summary: Calc pnl. Also hav the option to supply the profit vec directly

    if profit_1d2d is None: profit_1d2d = profit_vec(y_est_1d2d, y_true_1d2d)
    
    return numpy.sum(profit_1d2d)

def max_drawdown(v_1d):
    return numpy.max(numpy.maximum.accumulate(v_1d) - v_1d)

def ppt(y_est_1d2d, y_true_1d2d, profit_1d2d=None):
    # Summary: Calc ppt. Also hav the option to supply the profit vec directly

    if profit_1d2d is None: profit_1d2d = profit_vec(y_est_1d2d, y_true_1d2d)
    if numpy.all(profit_1d2d == 0): return 0 # if no trades were made.
    
    return numpy.sum(profit_1d2d) / float(numpy.count_nonzero(profit_1d2d != 0))

def hitrate(y_est_1d2d, y_true_1d2d, profit_1d2d=None):
    # Summary: Calc hitrate. Also hav the option to supply the profit vec directly

    if profit_1d2d is None: profit_1d2d = profit_vec(y_est_1d2d, y_true_1d2d)
    if numpy.all(profit_1d2d == 0): return 0.5 # if no trades were made.

    return numpy.count_nonzero(profit_1d2d > 0) / float(numpy.count_nonzero(profit_1d2d != 0))

def profit_vec(y_est_1d2d, y_true_1d2d):
    # summary: convert est and true to a profit vec.
    y_est_1d2d, y_true_1d2d = numpy.array(y_est_1d2d), numpy.array(y_true_1d2d)
    if (type(y_est_1d2d) != type(numpy.array([]))) or (type(y_true_1d2d) != type(numpy.array([]))): raise Exception("only numpy arrays.")
    if y_est_1d2d.shape != y_true_1d2d.shape: raise Exception("estimation and true vectors should share length.")
    if numpy.isnan(y_est_1d2d).any() or numpy.isnan(y_true_1d2d).any(): raise Exception("Array contained NaN value.")

    profit_1d2d = numpy.sign(y_est_1d2d) * y_true_1d2d

    return profit_1d2d

def graph_cumsum(y_est_1d=None, y_true_1d=None, profit_1d=None):
    if profit_1d is None: profit_1d = numpy.sign(y_est_1d) * y_true_1d
    plot_col_on_col(numpy.cumsum(profit_1d), plot_type="plot")

def rmse(y_est_1d, y_true_1d):
    return sqrt(mean_squared_error(y_est_1d, y_true_1d))

def mae(y_est_1d, y_true_1d):
    return abs(y_est_1d - y_true_1d).mean()

def mae_zero(y_est_1d, y_true_1d, ignore_zero=False):
    # summary: calculate mae, and m

    y_est_1d, y_true_1d = numpy.array(y_est_1d), numpy.array(y_true_1d)

    # whether it did saf or not to the y_est but did not set ignore_zero=True
    if ((numpy.count_nonzero(y_est_1d==0) / len(y_est_1d)) > .2) and (ignore_zero==False): raise Exception("Consider turning ignore zero to True")

    if ignore_zero: y_est_1d, y_true_1d = y_est_1d[y_est_1d!=0], y_true_1d[y_est_1d!=0]
    zero_vector_mae = mae(numpy.zeros(y_true_1d.shape), y_true_1d)
    actual_mae = mae(y_est_1d, y_true_1d)

    return zero_vector_mae - actual_mae

def custom_eval(y_est_1d, y_true_1d, eval_pow=0.5):
    # summary: calculate error by variable. from extremely high error hating (3) or extremely outliers robust (0.1).

    y_est_1d = (np.abs(y_est_1d) ** eval_pow) * np.sign(y_est_1d)
    y_true_1d = (np.abs(y_true_1d) ** eval_pow) * np.sign(y_true_1d)

    return abs(y_est_1d - y_true_1d).mean() ** (1.0/eval_pow)

def bin_classif_hitrate(y_est_1d, y_true_1d):
    """
    Hitrate for binary classification where the estimation is a value between [0,1] that is the probability estimation is 1. 
    And true is either 0 or 1 depends on classification task.
    """

    y_est_1d, y_true_1d = y_est_1d.copy(), y_true_1d.copy()

    # check this is a binary classifciation task
    if len(numpy.unique(y_true_1d)) != 2:
        raise

    # convert in true to (0,1)=>(-1,1) and in estimation all those below 0.5 are considered negative.
    y_true_1d[y_true_1d==0] = numpy.ones(len(y_true_1d[y_true_1d==0])) * -1
    y_est_1d[y_est_1d<0.5] = y_est_1d[y_est_1d<0.5] * -1

    return hitrate(y_est_1d, y_true_1d) 

def corr(v1, v2):
    if len(v1) != len(v2): raise Exception("Arrays do not share same length.")

    v1, v2 = numpy.array(v1).astype(float), numpy.array(v2).astype(float)

    if (v1 == v1[0]).all() or (v2 == v2[0]).all():
        return 0

    # pearsonr faster than numpy's corrcoef
    correlation = pearsonr(v1, v2)[0] #numpy.corrcoef(v1, v2)[0][1]

    return correlation

def corr_pval(v1, v2):
    if len(v1) != len(v2): raise Exception("Arrays do not share same length.")

    # if 2d but really 1d (50,1) then convert to (50). otherwise if not 1d hen exit.
    if (len(v1.shape) == 2) and (v1.shape[1] == 1): v1 = v1.reshape(-1)
    if (len(v2.shape) == 2) and (v2.shape[1] == 1): v2 = v2.reshape(-1)
    if (len(v1.shape) != 1) or (len(v2.shape) != 1): raise Exception("Only support 1dim.")


    corr, pval = pearsonr(v1, v2)
    
    if pval==0: return 1e10
    
    pval = 1.0/pval
    return pval

def corr_2_matrixes(x_2d=None, y_2d=None):
    """
    return corr coef between two groups of matrixes. Columns are the vectors. 
    The returned value at pos 7,3 means the correlation between column 7 of x and column 3 of y.
    corrcoef_2d[-1,-5] = corr(x_2d.values[:, -1], y_2d.values[:, -5])
    """
    
    if is_dataframe(x_2d): x_2d = x_2d.values
    if is_dataframe(y_2d): y_2d = y_2d.values
    if (not is_array(x_2d)) or (not is_array(y_2d)): raise Exception("Expected dataframe/array")

    xy_2d = array_append(x_2d, y_2d, axis=1)
    corrcoef_2d = numpy.corrcoef(xy_2d, rowvar=False)

    # trim (it calculates also all the 2600x2600 and 9x9 that are redundant, and also does everything twice)
    num_y_vectors, num_x_vectors = y_2d.shape[1], x_2d.shape[1]
    corrcoef_2d = corrcoef_2d[:num_x_vectors, -num_y_vectors:]
    
    return corrcoef_2d

def naive_saf(y_est_1d, lower_saf, upper_saf=100, include_zero=True):
    if type(y_est_1d) != type(numpy.array([])): raise Exception("Expects numpy array")
    if numpy.count_nonzero(y_est_1d) == 0: return y_est_1d

    new_y_est_1d = y_est_1d.copy()
    if include_zero:
        new_y_est_1d[abs(new_y_est_1d) < numpy.percentile(abs(new_y_est_1d), lower_saf)] = 0
        new_y_est_1d[abs(new_y_est_1d) > numpy.percentile(abs(new_y_est_1d), upper_saf)] = 0
    else:
        new_y_est_1d[abs(new_y_est_1d) < numpy.percentile(abs(new_y_est_1d[new_y_est_1d!=0]), lower_saf)] = 0
        new_y_est_1d[abs(new_y_est_1d) > numpy.percentile(abs(new_y_est_1d[new_y_est_1d!=0]), upper_saf)] = 0

    if (numpy.count_nonzero(y_est_1d) > 10) and (numpy.count_nonzero(y_est_1d) == numpy.count_nonzero(new_y_est_1d)): warnings.warn("Didn't zerofy any estimations, probably because many of them were already zerofied.")

    return new_y_est_1d

# single vector (supports 2d and 1d). 2d vectors are calculated per column (transpose if want row)
def eval_1d_funcs():
    # notes scores aren't to be removed just were annoying for me currently that's all.
    core_eval_funcs = {
        "min" : lambda v_2d: v_2d.min(axis=0),
        "max" : lambda v_2d: v_2d.max(axis=0),
        "median" : lambda v_2d: numpy.median(v_2d, axis=0),
        "sign" : lambda v_2d: numpy.sum(numpy.sign(v_2d), axis=0),
        "mean" : lambda v_2d: v_2d.mean(axis=0),
        "3rd_moment" : lambda v_2d: scipy.stats.moment(v_2d, moment=3),
        "4th_moment" : lambda v_2d: scipy.stats.moment(v_2d, moment=4),
        "mean_abs" : lambda v_2d: abs(v_2d).mean(axis=0),
        "std" : lambda v_2d: v_2d.std(axis=0),
        "std/mean" : lambda v_2d: (v_2d / abs(v_2d).mean(axis=0)).std(axis=0),
        "min_max" : lambda v_2d: v_2d.max(axis=0) - v_2d.min(axis=0),
        "min_max_mean" : lambda v_2d: (v_2d.max(axis=0) - v_2d.min(axis=0)) / abs(v_2d).mean(axis=0),
        "sign_sum_ratio" : lambda v_2d: _sign_sum_ratio(v_2d),
        "sum" : lambda v_2d: numpy.sum(v_2d, axis=0),
        "sum_abs" : lambda v_2d: abs(v_2d).sum(axis=0), 
        "sum/abs_sum" : lambda v_2d: numpy.sum(v_2d / abs(v_2d).mean(axis=0), axis=0),
    }

    all_eval_funcs = core_eval_funcs.copy()

    # important to use func_name as default argument as else lambda wouldn't use same value every time!
    for func_name in core_eval_funcs.keys():
        all_eval_funcs["neg_%s" % (func_name, )] = lambda v_2d, func_name=func_name: (core_eval_funcs[func_name](v_2d) * -1)

    for func_name in core_eval_funcs.keys():
        all_eval_funcs["abs_%s" % (func_name, )] = lambda v_2d, func_name=func_name: abs(core_eval_funcs[func_name](v_2d))

    for func_name in core_eval_funcs.keys():
        all_eval_funcs["abs_neg_%s" % (func_name, )] = lambda v_2d, func_name=func_name: abs(core_eval_funcs[func_name](v_2d)) * -1

    return all_eval_funcs
EVAL_VECTOR_FUNCS = eval_1d_funcs()

def _sign_sum_ratio(v_2d):
    # summary: a rather complex score function for zerofy in build. Calculates the ratio between sum of positive values and negatives in vector

    v_2d = v_2d.astype(float)

    # sums
    positives_sums_1d = abs(((v_2d > 0) * v_2d).sum(0))
    negatives_sums_1d = abs(((v_2d < 0) * v_2d).sum(0))

    if len(v_2d.shape) == ONE_DIM:
        return max(positives_sums_1d, negatives_sums_1d)
    elif len(v_2d.shape) == TWO_DIM:
        # if negative is 0 - it's a good sign, give a very small number just so not endless though
        negatives_sums_1d[negatives_sums_1d == 0], positives_sums_1d[positives_sums_1d == 0] = 1e-3, 1e-3

        # calculate ratios and return the larger one.
        positive_negative_ratio_1d = positives_sums_1d / negatives_sums_1d
        negative_positive_ratio_1d = negatives_sums_1d / positives_sums_1d
        scores_1d = numpy.maximum.reduce([positive_negative_ratio_1d, negative_positive_ratio_1d])
    else:
        raise Exception("Some weirdass shit 3d or something")

    return scores_1d

def autocorr(v1):
    # summary: calc auto correlation

    v1 = numpy.array(v1)
    return corr(v1[:-1], v1[1:])    

####### GRAPHS ######## 

def naive_safs_per_percentile(y_est_1d, y_true_1d, num_slices=10, num_relevant_slices=10, do_graph=False, do_print=True, use_values=False, file_path=""):
    # summary: graph the zerofy threshold with 10 diff quantiles. should be monotonic upwards if the zerofication func is meaningful

    if numpy.count_nonzero(y_est_1d) / float(len(y_est_1d)) < 0.9: raise Exception("May very well already have safs applied.")
    if num_slices <= 1: raise Exception("Weird num slices")
    if (num_relevant_slices is None) or (num_relevant_slices > num_slices): num_relevant_slices = num_slices

    # calc stepsize
    step_size = int(float(y_est_1d.size) / (num_slices+1))
    if step_size <= 0: step_size = 1

    # calc indexes
    slices_start_indexes = numpy.cumsum(numpy.array([0] + ([step_size] * num_relevant_slices)))
    slices_indexes = zip(slices_start_indexes[:-1], slices_start_indexes[1:])
    
    # score per slice
    quantile_scores, quantile_est_values = None, None
    my_y_est_1d = abs(y_est_1d.copy())
    est_sorted_indexes_1d = my_y_est_1d.flatten().argsort()[::-1]
    for start_i, end_i in slices_indexes:
        # score
        relevant_indexes = est_sorted_indexes_1d[start_i:end_i]
        cur_y_est_1d, cur_y_true_1d = y_est_1d.flatten()[relevant_indexes], y_true_1d.flatten()[relevant_indexes]
        cur_score = ppt(cur_y_est_1d, cur_y_true_1d) 
        # index
        val_of_bottom_est = my_y_est_1d.flatten()[est_sorted_indexes_1d[end_i]]
        # append
        quantile_est_values, quantile_scores = array_append(quantile_est_values, val_of_bottom_est), array_append(quantile_scores, cur_score)

    if do_print: 
        for score, val in zip(quantile_scores, quantile_est_values)[::-1]:
            print ("%.2f -> %.4f" % (val, score))
    if do_graph: plot_col_on_col(quantile_scores, quantile_est_values)
    if file_path != "": plot_col_on_col(quantile_scores, quantile_est_values, file_path=file_path)

def plot_col_on_col(y_1d, x_1d=None, title="graphy-graph", plot_type="scatter", file_path=""):
    # summary: plot col (vector) as function of another vector. If no x vector is given then just use indexes.
    # example: plot_col_on_col((y_train_pred_1d - y_train_1d) ** 2)
    # example: y_train_pred_1d = my_xgb.build_train(); err_1d = (y_train_pred_1d - y_train_1d) ** 2; print plot_col_on_col(err_1d, x_1d=y_train_pred_1d, title="err_pred")

    # just use indexes if x is not given
    if type(x_1d) == type(None): x_1d = pd.Series(range(len(y_1d)))

    # def func on click
    collected_dots = []
    def on_plot_click(event):
        indexes_1d = x_1d.iloc[event.ind.tolist()].index.values
        print (indexes_1d); collected_dots.append(indexes_1d)

    # plot
    fig = plt.figure(); fig.suptitle(title, fontsize=20); fig.canvas.mpl_connect('pick_event', on_plot_click);
    if plot_type == "scatter":
        plt.scatter(x_1d, y_1d, picker=True) #plt.plot also possible
    if plot_type == "plot":
        plt.plot(x_1d, y_1d)

    # save file if path is given
    if file_path: plt.savefig(file_path, bbox_inches='tight')
    else: plt.show()

    # convert list of lists to list
    collected_dots = [item for sublist in collected_dots for item in sublist]

    return collected_dots

def plot_by_occurences(h_1d, file_path=""):
    # summary: plot graph of number of apearences for every value

    h_1d = h_1d[~pd.isnull(h_1d)]
    sns.distplot(h_1d)
    if file_path: sns.plt.savefig(file_path, bbox_inches='tight')
    else: plt.show()

def corr_heatmap(data_2d):
    if type(data_2d) == type(numpy.array([])): data_2d = pandas.DataFrame(data_2d)

    corrmat = data_2d.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    
    sns.heatmap(corrmat, square=True)
    plt.setp(plt.yticks()[1], rotation=0)
    plt.setp(plt.xticks()[1], rotation=90)
    plt.show()

####### IO ########    
UNIQUE_PICKLE_ARRAY_NAME = "array77752341"
UNIQUE_PICKLE_FILE_NAME = "array77752341"

def my_pickle(data, file_name=None):
    # summary: saves to desktop a dict of NUMPY arrays.
    # param data: data can either be a DICT of numpy arrays, or just a single data array

    # only dict of array are accepted to the pickle. name is weirdass so it wouldnt be entered by accident
    if type(data) == type(numpy.array([])): 
        data_dict = {UNIQUE_PICKLE_ARRAY_NAME : data}
    elif type(data)  == type({}):
        data_dict = data
    else: 
        raise Exception("Must be either dict of numpy array")

    # the six thing is just a way to access random dict value that would work in both python 2 & 3
    if type(six.next(six.itervalues(data_dict))) != type(numpy.array([])):
        raise Exception("Only dict of pure numpy arrays are supported.")
    
    if file_name is None: file_name = UNIQUE_PICKLE_FILE_NAME
    file_path = os.path.join(r"C:\Users\pc\Desktop", file_name)
    # scipy.io.savemat(file_path, data_dict)
    numpy.savez(file_path, **data_dict)

def my_unpickle(file_name=None, file_path=None):
    if file_path is not None:
        return numpy.load(file_path)
    elif file_name is not None:
        file_path = os.path.join('C:\\Users\\pc\\Desktop\\', file_name) + ".npz"
        return numpy.load(file_path)
    elif os.path.isfile(os.path.join('C:\\Users\\pc\\Desktop\\', UNIQUE_PICKLE_FILE_NAME) + ".npz"):
        file_path = os.path.join('C:\\Users\\pc\\Desktop\\', UNIQUE_PICKLE_FILE_NAME) + ".npz"
        return numpy.load(file_path)[UNIQUE_PICKLE_ARRAY_NAME]
    else:
        raise Exception("Both path and name are empty.")

def file_train_test_split(original_file_path, train_file_path, test_file_path, train_quantile=0.75):
    # summary: split the training file to two - train file and internal test file.

    # get test and train data
    all_data_2d = pd.read_csv(original_file_path)
    train_observations_1d = numpy.array(range(int(len(all_data_2d) * train_quantile)))
    test_observations_1d = numpy.array(range(len(train_observations_1d), int(len(all_data_2d))))
    train_2d, test_2d = all_data_2d.iloc[train_observations_1d], all_data_2d.iloc[test_observations_1d]
    # write new files
    train_2d.to_csv(train_file_path, index=False)
    test_2d.to_csv(test_file_path, index=False)

def write_results(y_est_1d, first_index, id_col_name, results_col_name, file_path=None):
    ### summary: write prediction for a sumission.txt file.
    ### example: kagglib.write_results(y_pred_1d, 1461, "Id", "SalePrice", OUTPUT_FILE % (time(), ))
    
    if file_path == None: file_path = r"C:\Users\User\Desktop\results_%s_%s_%s.csv" % (datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second)

    # create dataframe #
    results_pd = pd.DataFrame()
    # write index col
    results_pd[ id_col_name ] = range(first_index, first_index + len(y_pred_1d))
    # write y estimation 
    results_pd[ results_col_name ] = y_est_1d

    results_pd.to_csv(file_path, index=False)

####### OTHER #######
def dataframe_append(base_1d2d, addition_1d2d):
    # summary: append pandas dataframes.

    if (type(addition_1d2d) != type(pandas.DataFrame())): raise Exception("addition not a dataframe (%s)" % (addition_1d2d, ))
    if base_1d2d is None: return addition_1d2d.copy().reset_index(drop=True)
    if (type(base_1d2d) != type(pandas.DataFrame())): raise Exception("base is not a dataframe (%s)" % (type(base_1d2d), ))

    return pandas.concat([base_1d2d, addition_1d2d], ignore_index=True)

def array_append(base_1d2d, addition_1d2d, axis=0):
    # summary: numpy own append does not support empty array - this really simplifies many code bullets.
    
    # if intenger/float convert to numpy array
    if (type(addition_1d2d) == type(INTEGER)) or type(addition_1d2d) == type(FLOAT) or type(addition_1d2d) == numpy.float64:
        addition_1d2d = numpy.array([addition_1d2d])
    
    # if none return addition as new
    if base_1d2d is None:
        return addition_1d2d

    
    if (axis == 1) and (len(base_1d2d.shape) == 1):
        base_1d2d = base_1d2d.reshape(-1, 1)
    if (axis == 1) and (len(addition_1d2d.shape) == 1):
        addition_1d2d = addition_1d2d.reshape(-1, 1)
    
    return numpy.append(base_1d2d, addition_1d2d, axis=axis)

def shape_all(list_of_items):
    for t in list_of_items:
        try:
            print (numpy.array(t).shape, type(t)) #end=''
        except Exception:
            print (t.detach().numpy().shape, type(t))
    # print (list(map(lambda x: x.shape, list_of_items)))

def is_dataframe(m_2d):
    return type(m_2d) == type(pandas.DataFrame())

def is_array(m_2d):
    return type(m_2d) == type(numpy.array([]))

def sort_dict(dict, do_print=False, num_print=1e10, reverse=True):
    # summary: return keys of dictionary sorted by value (note: their reverse is different from my reverse.)

    sorted_keys = []    
    for value, key in sorted( ((v,k) for k,v in dict.iteritems()), reverse=reverse): 
        sorted_keys.append(key)
        if do_print and (len(sorted_keys) < num_print): print (key, value)

    return sorted_keys

def sort_series(df_1d, sort_vec_1d=None, sort_abs=False, ascending=False):
    # summary: sort 1d dataframe either by its own values, or by a given vector values.

    df_1d = pandas.DataFrame(df_1d)

    # add col to sort by, eithe given or just the 1d values
    if sort_vec_1d is None:
        df_1d["sort_col"] = abs(df_1d) if sort_abs else df_1d
    else:
        df_1d["sort_col"] = sort_vec_1d

    # sort
    sorted_df_1d = df_1d.sort_values('sort_col', ascending=ascending).drop('sort_col', axis=1)

    return sorted_df_1d

def gen_all_subsets(l):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """

    powerset = list(chain.from_iterable(combinations(l ,n) for n in range(len(l)+1)))
    # convert it from list of tuple to list of lists
    powerset = map(list, powerset)
    
    return powerset

def print_list(l):
    for m in l: print (m)

def print_array(arr_1d):
    s = ""
    for v in arr_1d:
        s += "%.1f " % (v, )
    print (s)

## My Implementation ## 
def run_knn_classifier(Xtrn, Ytrn, Xtst, Ks):
    # Input:
    # Xtrn: M-by-D training data matrix
    # Ytrn: M-by-1 label vector for Xtrn
    # Xtst: N-by-D test data matrix
    # Ks: L-by-1 vector of the numbers of nearest neighbours in Xtrn
    # Output:
    # Ypreds : N-by-L matrix of predicted labels for Xtst

    Ypreds = []

    for test_row in Xtst:

        # deduct cur test row from the training sample
        Xtrn_difference_from_test_row = Xtrn - test_row
        
        # square distances of each of the training points from the cur test point
        square_distances = numpy.sqrt(numpy.sum(Xtrn_difference_from_test_row ** 2, axis=1))
        
        # for each k
        y_pred_per_k = []
        for k in Ks:
            # find k nearest neighbors
            nearest_neighbours_indexes = numpy.argsort(square_distances)[:k]
            # get the label of these k neighbors
            labels_of_neared_neighbors = Ytrn[nearest_neighbours_indexes]
            # find most common label
            most_common_label_in_neiborhood = numpy.argmax(numpy.bincount(labels_of_neared_neighbors))
            # append most common label to current y vector for current k value.
            y_pred_per_k.append(most_common_label_in_neiborhood)

        # add current prediction vectors to the overall predictions
        Ypreds.append(y_pred_per_k)
    
    # convert fro list to matrix
    Ypreds = numpy.array(Ypreds).astype(numpy.uint8)

    return Ypreds

def run_gaussian_classifiers(Xtrn, Ytrn, Xtst, epsilon):
    # Input:
    # Xtrn: M-by-D training data matrix
    # Ctrn: M-by-1 label vector for Xtrn
    # Xtst: N-by-D test data matrix
    # epsilon: A scalar parameter for regularisation
    # Output:
    # Cpreds: N-by-1 matrix of predicted labels for Xtst
    # Ms: D-by-K matrix of mean vectors
    # Covs: D-by-D-by-K 3D array of covariance matrices

    Cpreds = []
    Ms = []
    Covs = []

    # training
    for c in range(10):
        # get samples in current class
        cur_class_images = Xtrn[Ytrn == c]
        
        # estimate parameters
        cur_class_mean = MyMean(cur_class_images)
        cur_class_cov_matrix = MyCov(cur_class_images)

        # add regularization to prevent instability of inverse
        cur_class_cov_matrix += numpy.identity(len(cur_class_cov_matrix)) * epsilon

        # save the parameters in a dictionary
        Ms.append(cur_class_mean)
        Covs.append(cur_class_cov_matrix)

    # predicting
    labels_liklihood = []
    for c in range(10):
        # extract parameters from training for 
        cur_class_mean, cur_class_cov_matrix = Ms[c], Covs[c]

        # calculate probability of given class
        P_c = numpy.count_nonzero(Ytrn == c) / float(len(Ytrn))
        
        # calculate log likliehood
        likliehoods = numpy.sum(((-0.5 * (Xtst - cur_class_mean).dot(numpy.linalg.inv(cur_class_cov_matrix))).T * (Xtst - cur_class_mean).T), axis=0) \
            - 0.5 * logdet(cur_class_cov_matrix) \
            + numpy.log(P_c)
        
        # append current label likelihoods for all samples
        labels_liklihood.append(likliehoods)

    # find most likely class for each sample
    Cpreds = numpy.argmax(numpy.array(labels_liklihood), axis=0)

    return Cpreds, Ms, Covs

def MySqDist(p,q):
    return numpy.sqrt(numpy.sum((p - q) ** 2))

def MyMean(X):
    """
    Calculate (equivalent to X.mean(axis=0))
    """
    return numpy.sum(X, axis=0) / float(len(X))

def MyCov(X):
    """
    Calculate covariance of matrix X
    """

    X_normalized = (X - MyMean(X)) 
    cov_matrix = X_normalized.T.dot(X_normalized) / (X_normalized.shape[0] - 1)

    return cov_matrix

def comp_pca(X):
    # Input:
    # X: N * D matrix
    # Output:
    # Evecs: D-by-D matrix (double) contains all eigenvectors as columns
    # NB: follow the Task 1.3 specifications on eigenvectors.
    # EVals: Eigenvalues in descending order, D x 1 vector
    # (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)
    #  TO-DO

    EVecs = []
    EVals = []

    # compute cov matrix
    cov_matrix = MyCov(X)
     
    # get eigen values and eigen vectors of cov matrix!
    eigenvalues, eigenvectors = numpy.linalg.eig(cov_matrix)
    # convert real from complex numbers.
    eigenvalues = numpy.real(eigenvalues)
    eigenvectors = numpy.real(eigenvectors)
    
    # sort by desending order
    eigenvector_descending_order = numpy.argsort(eigenvalues)[::-1]
    EVecs = eigenvectors[:, eigenvector_descending_order]
    EVals = eigenvalues[eigenvector_descending_order]

    return EVecs, EVals

def comp_confmat(Ytrues, Ypreds, k):
    # Input:
    # Ytrues : N-by-1 ground truth label vector
    # Ypreds : N-by-1 predicted label vector
    # Output:
    # CM : K-by-K confusion matrix, where CM(i,j) is 
    #      the number of samples whose target is the ith class 
    #      that was classified as j
    # acc : accuracy (i.e. correct classification rate)

    CM = numpy.zeros((k, k))
    acc = 0.0

    # calculate confusion matrix.
    num_correct_classifications = 0
    for t, p in zip(Ytrues, Ypreds):
        CM[t][p] += 1

        # count correct classification
        if t == p:
            num_correct_classifications += 1

    # calculate accuracy as the propotion of sample it got right.
    acc = num_correct_classifications / float(len(Ytrues))  

    return CM, acc

def run_mgcs(Xtrain, Ytrain, Xtest, epsilon, L):
    # Input:
    #   Xtrain : M-by-D training data matrix (double)
    #   Ytrain : M-by-1 label vector for Xtrain (uint8)
    #   Xtest  : N-by-D test data matrix (double)
    #   epsilon : A scalar parameter for regularisation (double)
    #   L      : scalar (integer) of the number of Gaussian distributions per class
    # Output:
    #  Ypreds : N-by-1 matrix of predicted labels for Xtest (integer)
    #  MMs     : (L*K)-by-D matrix of mean vectors (double)
    #  MCovs   : (L*K)-by-D-by-D 3D array of covariance matrices (double)

    Ypreds = []
    MMs = []
    MCovs = []

    # training
    for c in range(10):
        # get samples in current class
        cur_class_images = Xtrain[Ytrain == c]

        # cluster current class
        C, idx, SSE = my_kMeansClustering(cur_class_images, L, cur_class_images[:L])
        # iterate over all clusters
        for c_cluster in range(L):
            # filter class images by current cluster
            cur_c_cluster_images = cur_class_images[idx == c_cluster]
        
            # estimate parameters
            cur_class_mean = MyMean(cur_c_cluster_images)
            cur_class_cov_matrix = MyCov(cur_c_cluster_images)

            # add regularization to prevent instability of inverse
            cur_class_cov_matrix += numpy.identity(len(cur_class_cov_matrix)) * epsilon

            # save the parameters in a dictionary
            MMs.append(cur_class_mean)
            MCovs.append(cur_class_cov_matrix)

    # predicting
    labels_liklihood = []
    for c in range(10):
        # calculate probability of given class
        P_c = numpy.count_nonzero(Ytrain == c) / float(len(Ytrain))

        # iterate over all clusters
        for c_cluster in range(L):
            # extract parameters from training for 
            cur_class_mean, cur_class_cov_matrix = MMs[c*L + c_cluster], MCovs[c*L + c_cluster]

            # calculate log likliehood
            likliehoods = numpy.sum(((-0.5 * (Xtest - cur_class_mean).dot(numpy.linalg.inv(cur_class_cov_matrix))).T * (Xtest - cur_class_mean).T), axis=0) \
                - 0.5 * logdet(cur_class_cov_matrix) \
                + numpy.log(P_c)
            
            # append current label likelihoods for all samples
            labels_liklihood.append(likliehoods)

    # find most likely class for each sample
    Ypreds = numpy.argmax(numpy.array(labels_liklihood), axis=0) / L

    return Ypreds, numpy.array(MMs), numpy.array(MCovs)

def my_kMeansClustering(X, k, initialCentres, maxIter=500):
    # Input
    # X : N-by-D matrix (double) of input sample data
    # k : scalar (integer) - the number of clusters
    # initialCentres : k-by-D matrix (double) of initial cluster centres
    # maxIter  : scalar (integer) - the maximum number of iterations
    # Output
    # C   : k-by-D matrix (double) of cluster centres
    # idx : N-by-1 vector (integer) of cluster index table
    # SSE : (L+1)-by-1 vector (double) of sum-squared-errors
    
    num_iterations = 0
    C = initialCentres
    idx = numpy.zeros(len(X), dtype=int)
    SSE = []
    prev_SSE, cur_iteration_SSE = numpy.nan, numpy.nan
    # cluster until all clusters are stable
    while (num_iterations < maxIter) and (cur_iteration_SSE != prev_SSE):
        
        # iterate over all points in X and assign each to the closest center 
        for index, point in enumerate(X):
            distance_to_center = []
            # iterate over all centers
            for center in C:
                distance_to_center.append(MySqDist(point, center))
            # assign to closest center
            idx[index] = int(numpy.argmin(distance_to_center))

        # append current round sum square error
        prev_SSE = cur_iteration_SSE
        cur_iteration_SSE = sum_square_errors(X, idx, C)
        SSE.append(cur_iteration_SSE)

        # recenter the clusters
        for cluster_index in range(k):
            C[cluster_index] = MyMean(X[idx == cluster_index])
        
        # increate iteration number.
        num_iterations += 1
    
    # add last round's error
    SSE.append(sum_square_errors(X, idx, C))
    SSE = numpy.array(SSE)

    return C, idx, SSE

def sum_square_errors(X, idx, C):
    """
    Return sum of all errors for all points
    """

    sse = 0
    for index, point in enumerate(X):
        cur_center = C[idx[index]]
        sse += numpy.sum((point - cur_center) ** 2)

    return sse

def euclidean_distance_efficient(Xtrn, Xtst):
    """
    Summary: compute euclidean distance of two matrices efficiently.
    """

    # calculate square distance
    XX = numpy.sum(numpy.square(Xtrn.astype(numpy.float32)), axis=1)
    YY = numpy.sum(numpy.square(Xtst.astype(numpy.float32)), axis=1)

    # convert to NxN and MxM matrixes
    XX = numpy.repeat(XX.reshape(-1, 1), len(YY), axis=1)
    YY = numpy.repeat(YY.reshape(-1, 1), len(XX), axis=1)

    # get distance matrix. NxM (for every test sample in row, a list of all distances to all test samples)
    D = numpy.sqrt(XX + YY.T - 2 * Xtrn.dot(Xtst.T)).T
    
    # note: d[i,j] is euclidean idstance between X[i] and Y[j].. computed VERY efficiently.
    return D


####### DEPRECATED #######
def _get_stacking_folds_est(regr, train_folds_data, x_test_2d, build_params, x_test_2_2d=None):
    # summary: do kfold on the trainfold sof data, with every model also predict on x_test. there'll N estimations on test and 1 on train.

    y_train_est_1d, y_test_ests_2d, y_test_2_ests_2d, my_model = None, None, None, copy.deepcopy(regr)
    # predict
    for x_train_2d, y_train_1d, x_pred_2d, y_pred_1d in train_folds_data:
        # build (do bag_build if parameters are supplied)
        my_model.set_data(x_train_2d, y_train_1d)

        # add fold estimation to the overall train est
        y_train_est_1d = array_append(y_train_est_1d, my_model.build(x_pred_2d, **build_params), EXTEND_ROW)
        y_test_ests_2d = array_append(y_test_ests_2d, my_model.build(x_test_2d, **build_params), ADD_COLUMN)
        if x_test_2_2d is not None: y_test_2_ests_2d = array_append(y_test_2_ests_2d, my_model.build(x_test_2_2d, **build_params), ADD_COLUMN)

    return y_train_est_1d, y_test_ests_2d, y_test_2_ests_2d

def make_estimations_matrix(regr, x_train_2d, y_train_1d, x_test_2d, x_test_2_2d=None, est_iterations=100, num_folds=2, build_params={}, train_grouping_1d=None):
    # summary: run the build_train method several tims every time shuffling the seed and the data to make a matrix ontop which a stack model will be made.
    # param obs_grouping_1d: the id of the group every observation belongs to (so it would be shuffled together)

    # new matrixes!
    stacking_ests_train_2d, stacking_ests_test_2d, stacking_ests_test_2_2d = None, None, None
    # create estimations matrix    
    for _ in range(est_iterations):
        # shuffle
        shuffled_x_train_2d, shuffled_y_train_1d, shuffled_train_indices_1d = _shuffle_group_rows(x_train_2d, y_train_1d, train_grouping_1d)
        # split train data to N folds
        train_folds_data = _split_data_to_folds(num_folds, shuffled_x_train_2d, shuffled_y_train_1d)
        # get estimations. note there'll be N estimations for test and only 1 for train (as every folding subset in kfold makes predictio nfor entire test.)
        train_est_1d, test_ests_2d, test_ests_2_2d = _get_stacking_folds_est(regr, train_folds_data, x_test_2d, build_params, x_test_2_2d=x_test_2_2d)

        # sort by indices of observations
        train_est_1d = numpy.array(sort_series(train_est_1d, sort_vec_1d=shuffled_train_indices_1d, ascending=True)).reshape(-1, )

        # appendchec
        stacking_ests_train_2d = array_append(stacking_ests_train_2d, train_est_1d, ADD_COLUMN)
        stacking_ests_test_2d = array_append(stacking_ests_test_2d, test_ests_2d, ADD_COLUMN)
        stacking_ests_test_2_2d = array_append(stacking_ests_test_2_2d, test_ests_2_2d, ADD_COLUMN)

    return stacking_ests_train_2d, stacking_ests_test_2d, stacking_ests_test_2_2d

## algorithms ##
def comp_pca(X):
    """
    Summary: Compute pca of matrix.
        # Input:
        # X: N * D matrix
        # Output:
        # Evecs: D-by-D matrix (double) contains all eigenvectors as columns
        # NB: follow the Task 1.3 specifications on eigenvectors.
        # EVals: Eigenvalues in descending order, D x 1 vector
        # (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)

        ## applying the pca:
        # # convert to 2D via PCA
        # EVecs, EVals = comp_pca(Xtrain)
        # Xtrain = Xtrain.dot(EVecs[:, :2])
    """

    EVecs = []
    EVals = []

    # compute cov matrix
    cov_matrix = MyCov(X)
     
    # get eigen values and eigen vectors of cov matrix!
    eigenvalues, eigenvectors = numpy.linalg.eig(cov_matrix)
    # convert real from complex numbers.
    eigenvalues = numpy.real(eigenvalues)
    eigenvectors = numpy.real(eigenvectors)
    
    # sort by desending order
    eigenvector_descending_order = numpy.argsort(eigenvalues)[::-1]
    EVecs = eigenvectors[:, eigenvector_descending_order]
    EVals = eigenvalues[eigenvector_descending_order]

    return EVecs, EVals

def kMeansClustering(X, k, initialCentres, maxIter=500):
    """
    Summary: kMeans clustering algorithm
        # Input
        # X : N-by-D matrix (double) of input sample data
        # k : scalar (integer) - the number of clusters
        # initialCentres : k-by-D matrix (double) of initial cluster centres
        # maxIter  : scalar (integer) - the maximum number of iterations
        # Output
        # C   : k-by-D matrix (double) of cluster centres
        # idx : N-by-1 vector (integer) of cluster index table
        # SSE : (L+1)-by-1 vector (double) of sum-squared-errors
    """

        num_iterations = 0
        C = initialCentres
        idx = numpy.zeros(len(X), dtype=int)
        SSE = []
        prev_SSE, cur_iteration_SSE = numpy.nan, numpy.nan
        # cluster until all clusters are stable
        while (num_iterations < maxIter) and (cur_iteration_SSE != prev_SSE):
            
            # iterate over all points in X and assign each to the closest center 
            for index, point in enumerate(X):
                distance_to_center = []
                # iterate over all centers
                for center in C:
                    distance_to_center.append(MySqDist(point, center))
                # assign to closest center
                idx[index] = int(numpy.argmin(distance_to_center))

            # append current round sum square error
            prev_SSE = cur_iteration_SSE
            cur_iteration_SSE = sum_square_errors(X, idx, C)
            SSE.append(cur_iteration_SSE)

            # recenter the clusters
            for cluster_index in range(k):
                C[cluster_index] = MyMean(X[idx == cluster_index])
            
            # increate iteration number.
            num_iterations += 1
        
        # add last round's error
        SSE.append(sum_square_errors(X, idx, C))
        SSE = numpy.array(SSE)

        return C, idx, SSE

def advanced_guassian(Xtrain, Ytrain, Xtest, epsilon, L):
    """
    Summary: GMM algorithm
    # Input:
    #   Xtrain : M-by-D training data matrix (double)
    #   Ytrain : M-by-1 label vector for Xtrain (uint8)
    #   Xtest  : N-by-D test data matrix (double)
    #   epsilon : A scalar parameter for regularisation (double)
    #   L      : scalar (integer) of the number of Gaussian distributions per class
    # Output:
    #  Ypreds : N-by-1 matrix of predicted labels for Xtest (integer)
    #  MMs     : (L*K)-by-D matrix of mean vectors (double)
    #  MCovs   : (L*K)-by-D-by-D 3D array of covariance matrices (double)
    """

    Ypreds = []
    MMs = []
    MCovs = []

    # training
    for c in range(10):
        # get samples in current class
        cur_class_images = Xtrain[Ytrain == c]

        # cluster current class
        C, idx, SSE = my_kMeansClustering(cur_class_images, L, cur_class_images[:L])
        # iterate over all clusters
        for c_cluster in range(L):
            # filter class images by current cluster
            cur_c_cluster_images = cur_class_images[idx == c_cluster]
        
            # estimate parameters
            cur_class_mean = MyMean(cur_c_cluster_images)
            cur_class_cov_matrix = MyCov(cur_c_cluster_images)

            # add regularization to prevent instability of inverse
            cur_class_cov_matrix += numpy.identity(len(cur_class_cov_matrix)) * epsilon

            # save the parameters in a dictionary
            MMs.append(cur_class_mean)
            MCovs.append(cur_class_cov_matrix)

    # predicting
    labels_liklihood = []
    for c in range(10):
        # calculate probability of given class
        P_c = numpy.count_nonzero(Ytrain == c) / float(len(Ytrain))

        # iterate over all clusters
        for c_cluster in range(L):
            # extract parameters from training for 
            cur_class_mean, cur_class_cov_matrix = MMs[c*L + c_cluster], MCovs[c*L + c_cluster]

            # calculate log likliehood
            likliehoods = numpy.sum(((-0.5 * (Xtest - cur_class_mean).dot(numpy.linalg.inv(cur_class_cov_matrix))).T * (Xtest - cur_class_mean).T), axis=0) \
                - 0.5 * logdet(cur_class_cov_matrix) \
                + numpy.log(P_c)
            
            # append current label likelihoods for all samples
            labels_liklihood.append(likliehoods)

    # find most likely class for each sample
    Ypreds = numpy.argmax(numpy.array(labels_liklihood), axis=0) / L

    return Ypreds, numpy.array(MMs), numpy.array(MCovs)

# ensembling techniques
def quantile_ensemble_conversion(stacking_2d, quantiles=[10,30,50,70,90]):
    # summary: in stacking, take the matrix of all estimations and sort every observation by value, then take as fts the feature that is at the N% quantiles as supplied

    # sort observation (rows) by value
    stacking_2d.sort(axis=ROWS)
    # get the num of cols that match every quantile
    col_nums = map(lambda x: int(x / 100.0 * stacking_2d.shape[COL_NUM]), quantiles)
    # convert to dataframe with feature name
    quantile_col_names = map(lambda x: "q_%s" % (x, ), quantiles)
    named_matrix_2d = pd.DataFrame(stacking_2d[:, col_nums], columns=quantile_col_names)
    
    return named_matrix_2d

def metrics_ensemble_conversion(stacking_x_2d):
    # summary: convert 2d array of predictions to a 2d matrix with mean, std, kurtosis, etc - many different v1 evaluations.

    print ("don't forget to add col names and make it dataframe.")
    pass

def zerofy_bad_obs(stacking_y_preds_2d, score_func, zerofy_quantile):
    ## summary: zerofy observations judged as bad according to the func
    # param score_func: score by which every observations is measured. Higher value=better

    # find bad observations
    observation_scores_1d = score_func(stacking_y_preds_2d.T)
    score_saf = numpy.percentile(observation_scores_1d, zerofy_quantile * FRACTION_TO_PERCENTAGE)
    bad_observations_1d = observation_scores_1d < score_saf

    if type(observation_scores_1d) != type(numpy.array([])) or len(observation_scores_1d.shape) != 1: raise Exception("Function must get 2d and return 1d")

    # zerofy bad observations
    zerofied_y_pred_1d = numpy.copy(stacking_y_preds_2d.mean(ROWS))
    zerofied_y_pred_1d[bad_observations_1d] = 0

    return zerofied_y_pred_1d

def data_sections_split(num_sections, x_train_2d, y_train_1d):
    # summary: split data to a list sections with equal lengths.
    section_size = int(len(x_train_2d) / num_sections)
    sections = []

    for index in range(num_sections):
        p_start = section_size * index
        p_end = p_start + section_size
        sections.append((x_train_2d[p_start:p_end], y_train_1d[p_start:p_end]))

    return sections

def fts_impotance_per_section(num_sections=3):
    ### summary: for each section, print lasso, corr and XGB fts importance. look for stability.
    pass

def find_best_linear(x_train_2d, y_train_1d, random_state=0, eval_params={}):
    print ("add elastic net")
    print ("add normal linear regression, especially to improve intuition")
    print ("add ransac based on lass or ridge or l1...")
    print ("can also try ensembling them..")
    linear_models = {
        "RANSAC": sklearn_wrapper(LIN_MODELS["RANSAC"](residual_threshold=1, min_samples=0.7, random_state=random_state, max_trials=7), x_train_2d, y_train_1d, random_seed=random_state), 
        "LASSO": sklearn_wrapper(LIN_MODELS["LASSO"](alpha=5e-2), x_train_2d, y_train_1d, random_seed=random_state),
        "L1": sklearn_wrapper(LIN_MODELS["L1"](epsilon=1), x_train_2d, y_train_1d, random_seed=random_state),
        "RIDGE": sklearn_wrapper(LIN_MODELS["RIDGE"](alpha=1e-2), x_train_2d, y_train_1d, random_seed=random_state),
        }

    if len(linear_models) != len(LIN_MODELS): raise Exception("There are more linear models ot use")

    for regr_name in linear_models.keys():
        print (regr_name,)
        eval_model(linear_models[regr_name], **eval_params)

    # todo: add ensemble.

def find_log_fts(lin_regr, x_train_2d, y_train_1d, threshold=-0.3, values_print=False):
    ### summary: finds which features to log and which to keep. Recommended to run twice to find fts to do log(log())
    ### example: kagglib.find_log_fts(regr, x_train_2d, y_train_1d, threshold=0.01)

    print ("needs to be rewritten using my wrapper...")
    print ("is log really the thing? maybe make it more generalized with normalize function. Otherwise there are other methods to do this as described here https://freedom89.github.io/Allstate_kaggle/")
    numeric_fts, results_dict = x_train_2d.dtypes[x_train_2d.dtypes != "object"].index, {}
    baseline_score = skl_eval_model(lin_regr, x_train_2d, y_train_1d, do_print=False)

    for ft in numeric_fts:
        # ignore dummy variables
        if len(x_train_2d[ft].unique()) <= 2: continue 
        # apply log & evaluate
        cur_x_train_2d = x_train_2d.copy(); cur_x_train_2d[ft] = numpy.log1p(cur_x_train_2d[ft])
        results_dict[ft] = baseline_score - skl_eval_model(lin_regr, cur_x_train_2d, y_train_1d, do_print=False)

    # print feature sorted by improvement to score
    if values_print:
        for ft_tuple in sorted(results_dict.items(), key=lambda x: x[1], reverse=True): print ("%-30s %.4f" % ft_tuple)
    
    # print the commaqnd to log features with value larger than threshold
    chosen_fts = [ft for ft, value in results_dict.items() if value >= threshold]
    print ("\n\nx_2d[%s] = numpy.log1p(x_2d[%s])" % (chosen_fts, chosen_fts, ))

def add_mul_fts(x_2d, bases_2d, targets_2d):
    # summary: auto engineer new linear features by multiplying bases and targets. return 2d DF with #_bases * #_tagets cols.
    # example: x_2d = add_mul_fts(x_2d, x_2d[["GarageArea"]], x_2d[["OverallQual", 'GarageQual', 'GarageCond']])

    new_fts_2d = pd.DataFrame()
    for base in bases_2d.columns:
        for target in targets_2d.columns:
            new_ft_name = "%s___%s" % (base, target, )
            new_fts_2d[new_ft_name] = bases_2d[base] * targets_2d[target]

    x_2d = pd.concat((x_2d, new_fts_2d), axis=1)
    return x_2d

def set_seed_all(seed=int(time() * 100) % 10**5, regr=None):

    random.seed(seed)
    numpy.random.seed(seed)

    if regr: regr.set_seed(seed)

def graph_zerofy_hitrates(y_est_1d, y_true_1d, quantiles=range(0, 100, 5)):
    ## summary: graph the zerofy threshold with 10 diff quantiles. should be monotonic upwards if the zerofication func is meaningful

    quantile_hitrates = None
    for zerofy_quantile in quantiles:
        # quantile_hitrates = array_append(quantile_hitrates, hitrate(zerofy_bad_obs(stacking_y_preds_2d, score_func, zerofy_quantile), y_test_1d))
        quantile_hitrates = array_append(quantile_hitrates, hitrate(naive_saf(y_est_1d, zerofy_quantile), y_true_1d))

    plot_col_on_col(quantile_hitrates, numpy.array(quantiles))


"""
truncated_v_2d = v_1d2d.copy()
        
        threshold_1d = truncated_v_2d.std(axis=0) * std_factor
        mean_1d = truncated_v_2d.mean(axis=0)
        print threshold_1d.shape, mean_1d.shape
        print (numpy.max(truncated_v_2d, axis=0))
        
        truncated_v_2d[truncated_v_2d > mean_1d + threshold_1d] = mean_1d + threshold_1d
        truncated_v_2d[truncated_v_2d < mean_1d - threshold_1d] = mean_1d - threshold_1d
        print (numpy.max(truncated_v_2d, axis=0))
        return truncated_v_2d

def analyze_vec(y_est_1d2d, y_true_1d2d, n=20, profit_1d2d=None):
    # Summary: Calc pnl. Also hav the option to supply the profit vec directly
    # param n: size on which stability is measured. 20 means roughly monthly, so that's the default

    if profit_1d2d is None: profit_1d2d = profit_vec(y_est_1d2d, y_true_1d2d) 
    vec_pnl, vec_ppt, vec_hitrate, vec_count = pnl(None, None, profit_1d2d=profit_1d2d), ppt(None, None, profit_1d2d=profit_1d2d), hitrate(None, None, profit_1d2d=profit_1d2d), numpy.count_nonzero(profit_1d2d != 0)

    # calc worst possible months
    sorted_profit_1d_2d = profit_1d2d[profit_1d2d.argsort()]
    monthly_min, monthly_max = numpy.sum(sorted_profit_1d_2d[:n]), numpy.sum(sorted_profit_1d_2d[-n:])
    
    # vec doesn't split exactly the section of length n, then remove the last few bits. truncate profit vec so that all months are of equal size exatly (ditch the last uneven month)
    num_of_months = len(profit_1d2d) / n
    shuffled_profit_1d2d = profit_1d2d.copy(); 
    if (len(shuffled_profit_1d2d) % num_of_months) != 0: shuffled_profit_1d2d = shuffled_profit_1d2d[:-(len(shuffled_profit_1d2d) % num_of_months)]
    # std monthly std and hitrate.    
    avg_monthly_std, avg_monthly_hitrate, iterations = 0, 0, 300
    for _ in range(iterations):
        numpy.random.shuffle(shuffled_profit_1d2d) # shuffle
        sections = numpy.split(shuffled_profit_1d2d, num_of_months) # split
        sections_profit_1d = numpy.array(map(lambda monthly_vec_1d: pnl(None, None, profit_1d2d=monthly_vec_1d), sections)) # pnl
        avg_monthly_std += sections_profit_1d.std(); avg_monthly_hitrate += hitrate(None, None, profit_1d2d=sections_profit_1d) # add stability measurements
    avg_monthly_std, avg_monthly_hitrate = avg_monthly_std/iterations, avg_monthly_hitrate/iterations

    return "Pnl: %.3f (%.3f*%.0f) <==> M.stability: %.3f/%.3f (%.1f-%.1f. %.3f)" % (vec_pnl, vec_ppt, vec_count, avg_monthly_hitrate, vec_hitrate, monthly_min, monthly_max, avg_monthly_std)


# def analyze_vec(v_1d2d, ignore_zero=True):
#     v_1d2d = numpy.array(v_1d2d)

#     if ignore_zero: v_1d2d = v_1d2d[v_1d2d!=0]
#     if v_1d2d.size == 0: return ""

#     return "%.3f (%.3f - %.3f, std: %.4f) - %.0f" % (numpy.mean(v_1d2d), numpy.min(v_1d2d), numpy.max(v_1d2d), numpy.std(v_1d2d), reduce(lambda x, y: x*y, v_1d2d.shape))

# def max_drawdown(profit_1d):
#     numpy.max(numpy.maximum.accumulate(profit_1d) - profit_1d)
"""