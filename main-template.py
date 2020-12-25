import sys; sys.path.append(r"C:\home\machine_learning")
import pandas, xgboost, numpy, kagglib, sklearn, operator, numpy as np, pandas as pd, random, os, datetime, sklearn, datetime; from time import time; from sklearn import linear_model; from kagglib import *; from kaggalgo import *; from scipy import stats; import numpy as np; import sys; sys.path.append(r"C:\home\git\Algo\research\ml_research\infra"); from initializer import identify_x_features

DEPTH=1; OBS_BAG=0.5; FTS_BAG=1; ETA=10e-2

def main():
	pass

def get_data():
	pandas.read_csv(test_file_path % (h,), comment="#").values
	x_2d = x_2d.astype(float)

XY_TRAIN = ""
X_PRED = ""
XGB_PARAMS = {
		'eta':ETA, # eta[0.01~]
		'max_depth':DEPTH, 'min_child_weight':0, #depth[2-7], min_child_weight[1-10]
		'subsample':OBS_BAG, 'colsample_bytree':FTS_BAG, #subsample[0.3-1], colsample_bytree[0.3-1]
		'gamma':0,
		
		'lambda':1, 'alpha':0,
		'max_delta_step':0,
		'scale_pos_weight':0, 

		'booster':'gbtree',
		'silent':1,
		'objective':'reg:linear',#'multi:softprob',#'reg:linear'
		# 'num_class':2,
		# 'eval_metric':bin_classif_hitrate,
		'seed':0
		}

if __name__ == "__main__":
	start = time()
	try:
		main()
	finally:
		print "\nTook: %.3f s" % (time() - start)