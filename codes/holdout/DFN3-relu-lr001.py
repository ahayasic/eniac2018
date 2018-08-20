from keras.models     import Sequential
from keras.layers     import Dense, Dropout
from keras.optimizers import Adagrad
from sklearn.model_selection import KFold
import keras.backend as K
import numpy	     as np
import argparse, os, time


def f1score(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	f1_score = (2*tp)/(2*tp+fp+fn+K.epsilon())

	return f1_score


def precision(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	fp = K.sum(y_neg * y_pred_pos)

	precision = tp/(tp+fp+K.epsilon())

	return precision


def sensitivity(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))

	tp = K.sum(y_pos * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	sensitivity = tp/(tp+fn+K.epsilon())

	return sensitivity


def specificity(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tn = K.sum(y_neg * y_pred_neg)
	fp = K.sum(y_neg * y_pred_pos)

	specificity = tn/(tn+fp+K.epsilon())

	return specificity


def print_scores(scores):
	print('--------------------------------------- training score ---------------------------------------')
	print('los: %.2f%% (+/- %.2f%%)\tacc: %.2f%% (+/- %.2f%%)\tsen: %.2f%% (+/- %.2f%%)\n'
	 	  'spc: %.2f%% (+/- %.2f%%)\tpre: %.2f%% (+/- %.2f%%)\tf1s: %.2f%% (+/- %.2f%%)' % 
	 	 ( np.mean(scores[6])*100,  np.std(scores[6])*100, 
		   np.mean(scores[7])*100,  np.std(scores[7])*100,
		   np.mean(scores[8])*100,  np.std(scores[8])*100,
		   np.mean(scores[9])*100,  np.std(scores[9])*100,
		   np.mean(scores[10])*100, np.std(scores[10])*100,
		   np.mean(scores[11])*100, np.std(scores[11])*100))

	print('-------------------------------------- validation score --------------------------------------')
	print('los: %.2f%% (+/- %.2f%%)\tacc: %.2f%% (+/- %.2f%%)\tsen: %.2f%% (+/- %.2f%%)\n'
	 	  'spc: %.2f%% (+/- %.2f%%)\tpre: %.2f%% (+/- %.2f%%)\tf1s: %.2f%% (+/- %.2f%%)' % 
	 	 ( np.mean(scores[0])*100, np.std(scores[0])*100, 
		   np.mean(scores[1])*100, np.std(scores[1])*100,
		   np.mean(scores[2])*100, np.std(scores[2])*100,
		   np.mean(scores[3])*100, np.std(scores[3])*100,
		   np.mean(scores[4])*100, np.std(scores[4])*100,
		   np.mean(scores[5])*100, np.std(scores[5])*100))


def load_data(input_file):
	print('Importing file ' + os.path.abspath(input_file))
	dataset = np.loadtxt(input_file, delimiter=',')
	X = dataset[:,0:340]
	Y = dataset[:,340]

	return X, Y


def main(args):
	np.random.seed(1)

	X, Y = load_data(args.input_file)
	model = Sequential()
	model.add(Dense(340, activation='relu',    kernel_initializer='he_uniform', input_dim=340))
	model.add(Dropout(0.5))
	model.add(Dense(340, activation='relu',    kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(340, activation='relu',    kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1,   activation='sigmoid', kernel_initializer='glorot_uniform'))
	
	adagrad = Adagrad(lr=0.01, decay=1e-9)	
	model.compile(optimizer = adagrad, 
				  loss      = 'mse',
				  metrics   = ['binary_accuracy', sensitivity, specificity, precision, f1score])
	
	history = model.fit(X, Y, validation_split=0.3, epochs=256, batch_size=128, verbose=0)
	scores  = list(history.history.values())
	print_scores(scores)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Insert the path to the input file')
	parser.add_argument('-in', action='store', dest='input_file', required=True, help='Input file in training set format.')

	start_time = time.time()
	args = parser.parse_args()
	print('Starting process!')
	main(args)
	minutes = (time.time() - start_time) / 60
	seconds = (time.time() - start_time) % 60
	print('Process completed!\n'
          'Time: %.0f minutes and %.d seconds' % (minutes, seconds))