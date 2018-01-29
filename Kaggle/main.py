import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from models import getModel
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from train_model import train_model_
import os 
from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit

def ShuffleS(X,y, splits = 1, train_s = 0.7):
	rs = ShuffleSplit(n_splits=splits, train_size=train_s, random_state=0)
	train_index, test_index = next((rs.split(X)))
	
	return X[train_index],X[test_index],y[train_index],y[test_index]
	
def StratifiedShuffleS(X,y, splits = 1, train_s= 0.7):
	rs = StratifiedShuffleSplit(n_splits=splits, random_state=0, train_size = train_s)
	train_index, test_index = next(rs.split(X,y))
	
	return X[train_index],X[test_index],y[train_index],y[test_index]
	
def GroupShuffleS_angle(X,y,angle, splits = 1,bins = [24.0, 34.0, 43.0, 53.0]):
	mask = (angle != 'na').values
	angle = angle.values[mask]
	
	x = angle.astype(float)
	groups = np.digitize(x, bins)
	
	X_without_na = X[mask,:,:,:]
	Y_without_na = y.values[mask]
	
	gss = GroupShuffleSplit(n_splits=splits, random_state=0)
	train_index, test_index = next(gss.split(X_without_na, Y_without_na, groups=groups))
	
	return X[train_index],X[test_index],y[train_index],y[test_index]

def GroupShuffleStratifiedS_angle(X,y,angle, splits = 1, bins = [24.0, 34.0, 43.0, 53.0]):
	
	mask = (angle != 'na').values
	angle = angle.values[mask]
	
	x = angle.astype(float)
	groups = np.digitize(x, bins)
	
	X_without_na = X[mask,:,:,:]
	Y_without_na = y.values[mask]
	
	gss = StratifiedShuffleSplit(n_splits=splits, random_state=0)
	train_index, test_index = next(gss.split(X_without_na,groups))
	
	return X[train_index],X[test_index],y[train_index],y[test_index]
	


def main():                                                                                                                           

	train = pd.read_json("train.json")
	#test = pd.read_json("test.json")


	X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
	X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
	X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
	
	target_train=train['is_iceberg']
	
	file_path = ".model_weights.hdf5"

	#X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, random_state=1, train_size=0.75)
	X_train_cv, X_valid, y_train_cv, y_valid = GroupShuffleStratifiedS_angle(X_train, target_train, train.inc_angle)
	
	
	#create and set model
	gmodel = getModel()
	mypotim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='binary_crossentropy', optimizer=mypotim, metrics=['accuracy'])

	#train our model
	gmodel = train_model_(gmodel, X_train_cv, y_train_cv, X_valid, y_valid, file_path)

	#download our best weights from file, evaluate our model on 25% train data 
	gmodel.load_weights(filepath=file_path)
	score = gmodel.evaluate(X_valid, y_valid, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	
	'''
	#prepare test data 
	X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
	X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
	X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
							  , X_band_test_2[:, :, :, np.newaxis]
							 , ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
							 
	#make prediction(probobility of existence of iceberg) 
	predicted_test=gmodel.predict_proba(X_test)

	#make submission
	submission = pd.DataFrame()
	submission['id']=test['id']
	submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
	submission.to_csv('sub.csv', index=False)
	'''
if __name__ == "__main__":
    main()
	
