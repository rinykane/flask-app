
import pandas as pd
import math
import numpy as np

features = ['crime_rate', 'avg_number_of_rooms','distance_to_employment_centers', 'property_tax_rate','pupil_teacher_ratio']


def normalize(x):
	# Compute x_norm as the norm 2 of x. 
    x_norm = np.linalg.norm(x,axis=0)

    # Divide x by its norm.
    x = np.divide(x, x_norm)
    return x, x_norm

def closest_ind(vlist, value, num):
    closest = vlist.loc[(vlist-value).abs().argsort()[:num]]
    return closest

def calc_dev(params, num):
	# Read housing data
	Housing=pd.read_csv('housing.csv')
	# Store values
	y = Housing['house_value']

	# Normalize values and store parameters
	normalized_housing, norm_params = normalize(Housing)
	# Add sum of normalized parameters
	normalized_housing['param_sum'] = normalized_housing[features].sum(axis=1)
	# Ensure that indexing is correct
	try:
		params = params.reindex(columns=features)
	except:
		print("Invalid parameters: "+ str(params))
		return 0
	# Normalize params and calulate sum of parameters
	norm_params = params/norm_params[0:5]
	s = (float) (norm_params.sum(axis=1))
	ind = closest_ind(normalized_housing['param_sum'], s, num)

	return(Housing['house_value'][ind.index].std())
