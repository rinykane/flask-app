
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
	new = pd.DataFrame()
	new = vlist

	# subtract values from data
	new = np.subtract(vlist,value)
	# sum all parameters together and find closest
	new['sum'] = new.sum(axis=1)
	closest = vlist.loc[(new['sum']).abs().argsort()[:num]].index 
	return closest

def calc_dev(params, num):
	# Read housing data
	Housing=pd.read_csv('housing.csv')
	# Store values
	y = Housing['house_value']

	# Normalize values and store parameters
	normalized_housing, norm_params = normalize(Housing)
	# Ensure that indexing is correct
	try:
		params = params.reindex(columns=features)
	except:
		return 0
	# Normalize params and calulate sum of parameters
	ref_params = params/norm_params[0:5]
	ind = closest_ind(normalized_housing[features], ref_params, num)

	return(Housing['house_value'][ind].std())
