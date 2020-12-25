from __future__ import print_function
from sklearn import linear_model
import numpy; import numpy as np
import kagglib
import statsmodels.api as sm

# import pandas as pd, numpy as np, seaborn as sns
# import pandas, numpy, sys
# from time import time
# from sklearn.model_selection import cross_val_score
# import matplotlib.pylab as plt
# import sklearn
# import seaborn as sns
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from collections import Counter
# import copy
# # import scipy.stats
# from itertools import chain, combinations
# import random
# # import os
# # import scipy.io
# # import pickle
# # from scipy.stats import mstats
import warnings
# import xgboost

# consts
PERCENTILE = 1
FUNC = 0
OBS_NUM = 0
COL_NUM = 1
ROWS = 1
ONE_DIM = 1
TWO_DIM = 2
BEST_CONFIG_POS = 0
P_VALUE = 1

####### ALGOS #######  

class Wrapper:
    # Summary: Base of all wrappers of internet algorithms.
    # Points:
    #    1. All datas are of numpy array type (not pandas! - except for specific get ft importance function).

    def _unnormalize_label(self, y_1d, y_std):
        return y_1d * y_std

    def _avg_seed_iterations(self, seed_iterations, x_pred_2d):
        # summary: ensemble by shuffling seed and averaging. Good to get variance&error down.

        overall_y_pred_1d, prev_pred_1d = numpy.zeros(len(x_pred_2d)), numpy.array([])
        # varage result over many iterations (each time with diff seed naturally)
        for i in range(seed_iterations):
            # change the model seed. this weird thing achives that it is dependent on seed (so same seed gives same result), and yet no overlapping (so seed=1 and seed=0 wont have too many overlapping executions)
            self.set_seed(seed=int(hash(str(self.random_seed+i))%1e9))
            # sum
            y_pred_1d = self._make_prediction(x_pred_2d)

            # print kagglib.hitrate(y_pred_1d, self.y_train_1d)
            overall_y_pred_1d += y_pred_1d

            # check iterations are meaningful
            self._test_iterations_meaningful(y_pred_1d, prev_pred_1d)
            prev_pred_1d = y_pred_1d

        self.set_seed() # return regr seed to normal
        # avg the sum of all predictions
        y_pred_1d = overall_y_pred_1d / seed_iterations

        return y_pred_1d

    def build(self, x_pred_2d, seed_iterations=1):
        # summary: build a model %s iterations times, every time with different %S random observations. Of those iterations, guess mean value on all observations whose variance is in the bottom %s variance quantile. variance (usually defined as pow 2) is also up for optimization (higher value - more significance for prediction outliers).
        # param bagging_iter: number of bagging iterations
        # param bagging_obs: percentage of features to be bagged as well in every iteration
        # param zerofy: Receives a code for the type of score (see _get_zerofy_rating) and percentile of observation to zerofy (e.g 0.5) and zerofys those observations.

        if self.random_seed == None: raise Exception("Very recommended to seed when using build!")
        
        # init
        x_train_2d, y_train_1d, x_pred_2d = self.x_train_2d, self.y_train_1d, numpy.array(x_pred_2d)
        
        # bagging & normalize if required
        y_pred_1d = self._avg_seed_iterations(seed_iterations, x_pred_2d)

        # if not classification, then check the prediction is distributed normally.
        if len(numpy.unique(y_train_1d)) > 10:
            self._test_pred_normal(y_pred_1d)
        
        return y_pred_1d

    def _get_folds_estimations(self, folds_data, build_params):
        # summary: receives folds data and return the estimaiton per fold

        overall_y_est_1d, original_data, fold_estimations = numpy.array([]), (self.x_train_2d, self.y_train_1d), []
        # predict
        for x_train_2d, y_train_1d, x_pred_2d, y_pred_1d in folds_data:
            # build (do bag_build if parameters are supplied)
            self.set_data(x_train_2d, y_train_1d)
            y_est_1d = self.build(x_pred_2d, **build_params)
            
            # for i in range(100):
            #     print (zip(x_pred_2d[i], [u'predictor_corr_sum', u'predicted_corr_sum_reversed']))
            #     print (y_est_1d[i], y_pred_1d[i])
            # exit()

            overall_y_est_1d = numpy.append(overall_y_est_1d, y_est_1d)
            # add score to fold to calcualte std
            fold_estimations.append((y_est_1d, y_pred_1d))

        # set back to original data.
        self.set_data(*original_data)

        return overall_y_est_1d, fold_estimations        

    def build_train(self, num_folds=2, build_params={}):
        # summary: split train to two and train on one part & predict on the other to get train predictions. Note: using build function would be preferable but is not very feasible cuz dmatrix is a dumb shit and hard to convert to numpy arrays
        # param normalize_post_outlers: whether to normalize after outliers removal (reduce mean from y)
        # example: y_train_pred_1d = regr.build_train(outliers=find_outliers_extremes_1d(y_train_1d, drop_quantile=1))
        # example: y_train_pred_1d = regr.build_train(bagged_params={"num_iter":10, "obs_bag":0.3, "zerofy_percentile":=0}) 

        folds_data = kagglib._split_data_to_folds(num_folds, self.x_train_2d, self.y_train_1d)

        # fold estimations are pairs of (estimation, true) per fold.
        overall_y_est_1d, fold_estimations = self._get_folds_estimations(folds_data, build_params)

        if len(overall_y_est_1d) != len(self.y_train_1d): raise Exception("should be same length!")

        return overall_y_est_1d, fold_estimations

    # tests
    def _test_pred_normal(self, y_pred_1d):
        ## summary: test y_pred is roughly normal
        
        # neg_sum = abs(float(sum(y_pred_1d[y_pred_1d < 0])))
        # pos_sum = float(sum(y_pred_1d[y_pred_1d > 0]))
        norm_y_pred_1d = y_pred_1d - y_pred_1d.mean()
        num_neg = float(numpy.count_nonzero(norm_y_pred_1d < 0))
        num_pos = float(numpy.count_nonzero(norm_y_pred_1d > 0))
        
        if (numpy.isclose(num_pos, 0) or numpy.isclose(num_neg, 0)):
            
            # kagglib.plot_by_occurences(y_pred_1d)
            if len(y_pred_1d)>1: 
                warnings.warn("Pred all zero")
                print (self.x_train_2d.shape)
                print (y_pred_1d)
                raise
            return
            
        
        neg_pos_ratio = num_neg / num_pos    
        if (len(y_pred_1d) > 200 and (neg_pos_ratio > 3 or neg_pos_ratio < 1/3.0)) or (len(y_pred_1d) < 200 and (neg_pos_ratio > 10 or neg_pos_ratio < 1/10.0)):
            # NOTE: this means a prediction had way more positive or way more negative values.
            # print self.y_train_1d[:100]
            # print y_pred_1d[:100]
            # print neg_pos_ratio
            # print len(y_pred_1d)
            # kagglib.plot_by_occurences(y_pred_1d)
            warnings.warn("very inbalanced neg/pos ratio")

    def _test_iterations_meaningful(self, y_pred_1d, prev_pred_1d):
        # NOTE: means that in the iterations in the build model, two predictions were the same.
        if numpy.array_equal(y_pred_1d, prev_pred_1d):
            print (y_pred_1d)
            print (prev_pred_1d)
            print (len(prev_pred_1d))
            warnings.warn("Seed iterations wasted.")
            raise

    ################################################################################################################

    def __init__(self, model, x_train_2d, y_train_1d, random_seed=None):
        raise NotImplementedError
    
    def _get_fts_importance(self, col_names):
        raise NotImplementedError

    def _make_prediction(self, x_pred_2d):
        raise NotImplementedError            

    def set_data(self, x_train_2d, y_train_1d):
        raise NotImplementedError

    def set_seed(self):
        raise NotImplementedError

    def set_param(self, name, value):
        raise NotImplementedError

class statsmodel_wrapper(Wrapper):
    def __init__(self, x_train_2d, y_train_1d, random_seed=0):
        """
        Based on Segei's line: [b,stat] = robustfit(lastdata(goodind,mai),trainingtargetsmod(good_ind,T),'bisquare',8);
        https://www.statsmodels.org/stable/rlm.html
        https://www.statsmodels.org/devel/generated/statsmodels.robust.robust_linear_model.RLM.html
        https://www.mathworks.com/help/stats/robustfit.html
        """

        if numpy.isclose(abs(x_train_2d).sum(axis=1), 0).any() and (x_train_2d.shape[1] >= 4): warnings.warn("There was an all-zeros sample in xy file data.")
        if numpy.isclose(abs(x_train_2d).sum(axis=0), 0).any(): warnings.warn("There was an all-zeros feature in xy file data.")
        self.set_data(numpy.array(x_train_2d), numpy.array(y_train_1d))
        if random_seed != None: self.set_seed(random_seed)

    def _make_prediction(self, x_pred_2d):
        ### summary: build a model and return predictoin on predict data
        # sets the model. add constant adds the intercept feature to the data        
        self.model = sm.RLM(self.y_train_1d, sm.add_constant(self.x_train_2d), M=sm.robust.norms.TukeyBiweight(c=5))
        fitted_model = self.model.fit() 
        # predicts. add constant adds the intercept feature to the data
        y_pred_1d = self.model.predict(params=fitted_model.params, exog=sm.add_constant(x_pred_2d))
        
        return y_pred_1d

    def set_data(self, x_train_2d, y_train_1d):
        if len(x_train_2d) != len(y_train_1d):
            raise Exception("Number of observations between x and y differ")

        # print x_train_2d.shape, y_train_1d.shape
        self.x_train_2d, self.y_train_1d = numpy.array(x_train_2d), numpy.array(y_train_1d)

    def set_seed(self, seed=None):
        if seed is not None: self.random_seed = seed

class sklearn_wrapper(Wrapper):
    def __init__(self, model, x_train_2d, y_train_1d, random_seed=None):
        # example: regr = sklearn_wrapper(linear_model.Lasso(max_iter=1e6, alpha=1), x_train_2d, y_train_1d, x_pred_2d)
        self.model = model

        self.set_data(numpy.array(x_train_2d), numpy.array(y_train_1d))

        if numpy.isclose(abs(x_train_2d).sum(axis=1), 0).any() and (x_train_2d.shape[1] > 2): warnings.warn("There was an all-zeros sample in xy file data.")
        if numpy.isclose(abs(x_train_2d).sum(axis=0), 0).any(): warnings.warn("There was an all-zeros feature in xy file data.")

        if random_seed != None: 
            self.set_seed(random_seed)

    def _get_fts_importance(self, col_names):
        ### Dataframe of importance values with indexes the ft name.

        # need feature names so only dataframes are allowed (not numpy arrays)
        # if type(x_train_2d) != type(pandas.DataFrame([])): raise Exception("Sorry man but only dataframes allowed as we need nameynames")

        self.model.fit(self.x_train_2d, self.y_train_1d)
        print (self.model.coef_)
        fts_importance_1d = pd.DataFrame(self.model.coef_, index=col_names)

        return fts_importance_1d

    def _make_prediction(self, x_pred_2d):
        ### summary: build a model and return predictoin on predict data
        ### example: y_pred_1d = kagglib.skl_build(regr, x_train_2d, y_train_1d, x_pred_2d)

        self.model.fit(numpy.array(self.x_train_2d), numpy.array(self.y_train_1d))
        y_pred_1d = self.model.predict(x_pred_2d)
        
        return y_pred_1d

        #### bagging
        # y_pred_1d = numpy.zeros(len(x_pred_2d))
        # OBS_BAG = 0.7; OBS_ITER=30

        # for i in range(OBS_ITER):
        #     indexes_choice_1d = np.random.choice(int(len(self.x_train_2d)), int(len(self.x_train_2d)*OBS_BAG), replace=False)
        #     # indexes_choice_1d.sort()
        #     temp_x_train_2d, temp_y_train_1d = self.x_train_2d[indexes_choice_1d], self.y_train_1d[indexes_choice_1d]
            
            
        #     self.model.fit(numpy.array(temp_x_train_2d), numpy.array(temp_y_train_1d))
        #     y_pred_1d += self.model.predict(x_pred_2d)
        
        # y_pred_1d = y_pred_1d/float(OBS_ITER)

        # return y_pred_1d

    def set_data(self, x_train_2d, y_train_1d):
        if len(x_train_2d) != len(y_train_1d):
            raise Exception("Number of observations between x and y differ")

        # print x_train_2d.shape, y_train_1d.shape
        self.x_train_2d, self.y_train_1d = numpy.array(x_train_2d), numpy.array(y_train_1d)

    def set_seed(self, seed=None):
        if seed is not None: self.random_seed = seed

        if self.model.__dict__.has_key("random_state"):
            # self.model.set_params(random_state=self.random_seed)
            self.model.__dict__["random_state"] = self.random_seed

    def set_param(self, name, value):
        # ransac alpha
        if (type(self.model) == LIN_MODELS["RANSAC"]) and (name == "alpha"):
            self.model.__dict__['base_estimator'].alpha = value
        
        self.model.__dict__[name] = value
    
    def get_param(self, name):
        # ransac alpha
        if (type(self.model) == LIN_MODELS["RANSAC"]) and (name == "alpha"):
            return self.model.__dict__['base_estimator'].alpha

        return self.model.__dict__[name]

class xgboost_wrapper(Wrapper):
    def __init__(self, x_train_2d, y_train_1d, nrounds, params=None, shuffle=True, custom_obj=None, cv_folds=5):
        if params:
            self.params = params
        else:
            self.params = {
                'eta':0.04, # eta[0.01~]
                'max_depth':3, 'min_child_weight':3, #depth[2-7], min_child_weight[1-10]
                'subsample':0.7, 'colsample_bytree':0.8, #subsample[0.3-1], colsample_bytree[0.3-1]
                
                'gamma':0, #[0.1-0.5] - maybe. didn't help me in the past.
                'lambda':1, 'alpha':0.3,
                'max_delta_step':0,
                'scale_pos_weight':0, 

                'booster':'gbtree',
                'silent':1,
                'objective':'reg:linear',
                'eval_metric':"rmse",
                'seed':0
                }

        # set seed
        self.set_seed(self.params['seed'])

        self.nrounds = nrounds
        self.set_data(x_train_2d, y_train_1d)

        # paramters for cross validation in eval_model function. They're here for tune_param to work smoothly.
        self.shuffle = shuffle
        self.cv_folds = cv_folds

        # custom objective function. usual reg:linear is RMSE sometime I want MAE, or other..
        self.custom_obj = xgboost_wrapper._fair_obj if custom_obj == "l1" else custom_obj

    def _get_fts_importance(self, col_names):
        ### summary: plot/prints feature importance according to an xgboost build. Dataframe of importance values with indexes the ft name.

        # get
        xgb_model = xgboost.train(self.params, xgboost.DMatrix(pandas.DataFrame(self.x_train_2d, columns=col_names), label=self.y_train_1d), num_boost_round=self.nrounds, obj=self.custom_obj)
        # sort
        fts_importance_dict = xgb_model.get_fscore()

        # xgb does not include fts that appeared 0 times.. change that to simplify ploting.
        for ft_name in col_names:
            if not fts_importance_dict.has_key(ft_name):
                fts_importance_dict[ft_name] = 0

        # dict to dataframe
        fts_importance_1d = pandas.DataFrame(fts_importance_dict.values(), index=fts_importance_dict.keys())

        return fts_importance_1d

    def _make_prediction(self, x_pred_2d):
        ### summary: build a model and return predictoin on predict data
        ### example: y_pred_1d = kagglib.skl_build(regr, x_train_2d, y_train_1d, x_pred_2d)

        xgb_model = xgboost.train(self.params, self.train_dmatrix, num_boost_round=self.nrounds, obj=self.custom_obj)
        y_pred_1d = xgb_model.predict(xgboost.DMatrix(x_pred_2d))
        # xgboost.plot_tree(xgb_model, num_trees=1); plt.show(); exit()

        return y_pred_1d

    def set_data(self, x_train_2d, y_train_1d):
        ### summary: set this wrappers data to use
        ### example: my_xgb.set_data(x_train_2d, y_train_1d, x_pred_2d)
        
        if len(x_train_2d) != len(y_train_1d):
            raise Exception("Number of observations between x and y differ")

        self.x_train_2d, self.y_train_1d = x_train_2d, y_train_1d
        self.train_dmatrix = xgboost.DMatrix(x_train_2d, label=y_train_1d)

    def set_seed(self, seed=None):
        if seed is not None: self.random_seed = seed
        self.params['seed'] = self.random_seed

    def set_param(self, name, value):
        self.params[name] = value

    def get_param(self, name):
        return self.params[name]

    def get_num_trees(self, verbose=False, custom_feval=None):
        ### summary: evaluate the xgb model on train set using CV, over several iterations, one to reduce variance and another to check variance. Also good because it gives the proper nround..

        # init
        my_params, start_time = dict(self.params), time()

        # get score
        cvresults = xgboost.cv(my_params, self.train_dmatrix, num_boost_round=3000, early_stopping_rounds=10, nfold=self.cv_folds, verbose_eval=True, feval=self._feval_wrapper(custom_feval), shuffle=self.shuffle, obj=self.custom_obj)

        # print
        print ("%0.4f (std: %0.4f) |  %s trees |  %s sec" % (cvresults.tail(1)[cvresults.columns[0]].values[0] , cvresults.tail(1)[cvresults.columns[0]].values[0] , len(cvresults), time() - start_time, ))
    
    def _feval_wrapper(self, my_metric):
        """
        feval wrapper - you give it a function it returns a function ready to be input as a custom feval to xgboost
        """

        # if feval is none just return none.
        if my_metric is None: return None

        # generate feval
        def my_feval(y_pred, y_true):
            y_true = y_true.get_label()

            # return hitrate, add *-1 for XGB because it likes to minimize error
            return my_metric.__name__, my_metric(y_pred, y_true) * -1
        
        return  my_feval
        
LIN_MODELS = {
    # regr = sklearn_wrapper(LIN_MODELS["RANSAC"](residual_threshold=2, min_samples=0.7, random_state=random_state, max_trials=7), x_train_2d, y_train_1d, random_seed=random_state) 
    "RANSAC" : linear_model.RANSACRegressor, # good method dealing with noisy dataset - finds a good way to deal with outliers.
    # regr = sklearn_wrapper(LIN_MODELS["LASSO"](alpha=1e-2), x_train_2d, y_train_1d, random_seed=random_state)
    "LASSO" : linear_model.Lasso,
    # regr = sklearn_wrapper(LIN_MODELS["L1"](epsilon=1), x_train_2d, y_train_1d, random_seed=random_state)
    "L1" : linear_model.HuberRegressor, # set epsilon to 1 for l1
    # regr = sklearn_wrapper(LIN_MODELS["RIDGE"](alpha=1e-2), x_train_2d, y_train_1d, random_seed=random_state)
    "RIDGE" : linear_model.Ridge, 
}

"""
#### bagging
        # y_pred_1d = numpy.zeros(len(x_pred_2d))
        # OBS_BAG = 0.7; OBS_ITER=30
        # # OBS_BAG = 1.0; OBS_ITER=1

        # for i in range(OBS_ITER):
        #     indexes_choice_1d = np.random.choice(int(len(self.x_train_2d)), int(len(self.x_train_2d)*OBS_BAG), replace=False)
        #     # indexes_choice_1d.sort()
        #     temp_x_train_2d, temp_y_train_1d = self.x_train_2d[indexes_choice_1d], self.y_train_1d[indexes_choice_1d]
            
        #     self.model = sm.RLM(self.y_train_1d, sm.add_constant(self.x_train_2d), M=sm.robust.norms.TukeyBiweight(c=5))
        #     fitted_model = self.model.fit() 
        #     # predicts. add constant adds the intercept feature to the data
        #     y_pred_1d += self.model.predict(params=fitted_model.params, exog=sm.add_constant(x_pred_2d))
        
        # y_pred_1d = y_pred_1d/float(OBS_ITER)

        # return y_pred_1d

self.model = linear_model.TheilSenRegressor(random_state=42)
"""