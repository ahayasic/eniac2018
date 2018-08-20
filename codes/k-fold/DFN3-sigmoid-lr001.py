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


def save_results(acc, sen, spc, pre, f1s):
	with open('results.txt', 'a') as file_results:
		file_results.write('-------------------------------- 10-Fold Cross Validation ------------------------------------\n')
		for i in range(0, len(acc)):
			file_results.write('Fold '   + str(i) +
							   '\tacc: ' + format(acc[i], '.2f') + '%\tsen: ' + format(sen[i], '.2f') + '%' + 
							   '\tspc: ' + format(spc[i], '.2f') + '%\tpre: ' + format(pre[i], '.2f') + '%' +
							   '\tf1s: ' + format(f1s[i], '.2f') + '%\n')
		
		file_results.write('----------- Mean -----------\n')
		file_results.write('acc: '  + format(np.mean(acc), '.2f') + '% (+/- ' + format(np.std(acc), '.2f') + '%)\n' +
						   'sen: '  + format(np.mean(sen), '.2f') + '% (+/- ' + format(np.std(sen), '.2f') + '%)\n' +
						   'spc: '  + format(np.mean(spc), '.2f') + '% (+/- ' + format(np.std(spc), '.2f') + '%)\n' +
						   'pre: '  + format(np.mean(pre), '.2f') + '% (+/- ' + format(np.std(pre), '.2f') + '%)\n' +
						   'f1s: '  + format(np.mean(f1s), '.2f') + '% (+/- ' + format(np.std(f1s), '.2f') + '%)\n')
		file_results.write('----------------------------------------------------------------------------------------------\n\n')


def load_data(input_file):
	print('Importing file ' + os.path.abspath(input_file))
	dataset = np.loadtxt(input_file, delimiter=',')
	X = dataset[:,0:340]
	Y = dataset[:,340]

	return X, Y


def main(args):
	np.random.seed(1)
	los = []
	acc = []
	sen = []
	spc = []
	pre = []
	f1s = []

	X, Y = load_data(args.input_file)
	kFold = KFold(n_splits=10)
	i = 0
	for train, test in kFold.split(X, Y):
		model = None
		model = Sequential()
		model.add(Dense(340, activation='sigmoid', kernel_initializer='glorot_uniform', input_dim=340))
		model.add(Dropout(0.5))
		model.add(Dense(340, activation='sigmoid', kernel_initializer='glorot_uniform'))
		model.add(Dropout(0.5))
		model.add(Dense(340, activation='sigmoid', kernel_initializer='glorot_uniform'))
		model.add(Dropout(0.5))
		model.add(Dense(1,   activation='sigmoid', kernel_initializer='glorot_uniform'))

		adagrad = Adagrad(lr=0.01, decay=1e-9)
		model.compile(optimizer = adagrad, 
					  loss      = 'mse',
					  metrics   = ['binary_accuracy', sensitivity, specificity, precision, f1score])

		history = model.fit(X[train], Y[train], epochs=256, batch_size=128, verbose=0)
		scores  = model.evaluate(X[test], Y[test], verbose=0)
		training_scores = list(history.history.values())
		print('-------------------------------------------- Fold ' + str(i) + ' ----------------------------------------')
		print('--------------------------------------- training score ---------------------------------------')
		print('los: %.2f%% (+/- %.2f%%)\tacc: %.2f%% (+/- %.2f%%)\tsen: %.2f%% (+/- %.2f%%)\n'
		 	  'spc: %.2f%% (+/- %.2f%%)\tpre: %.2f%% (+/- %.2f%%)\tf1s: %.2f%% (+/- %.2f%%)' % 
		 	 ( np.mean(training_scores[0])*100,  np.std(training_scores[0])*100, 
			   np.mean(training_scores[1])*100,  np.std(training_scores[1])*100,
			   np.mean(training_scores[2])*100,  np.std(training_scores[2])*100,
			   np.mean(training_scores[3])*100,  np.std(training_scores[3])*100,
			   np.mean(training_scores[4])*100,  np.std(training_scores[4])*100,
			   np.mean(training_scores[5])*100,  np.std(training_scores[5])*100))

		print('------------------------------------------ test score ----------------------------------------')
		print('los: %.2f%% (+/- %.2f%%)\tacc: %.2f%% (+/- %.2f%%)\tsen: %.2f%% (+/- %.2f%%)\n'
	 	  	  'spc: %.2f%% (+/- %.2f%%)\tpre: %.2f%% (+/- %.2f%%)\tf1s: %.2f%% (+/- %.2f%%)' % 
	 	 	 ( np.mean(scores[0])*100, np.std(scores[0])*100, 
		 	   np.mean(scores[1])*100, np.std(scores[1])*100,
		 	   np.mean(scores[2])*100, np.std(scores[2])*100,
		 	   np.mean(scores[3])*100, np.std(scores[3])*100,
		 	   np.mean(scores[4])*100, np.std(scores[4])*100,
		 	   np.mean(scores[5])*100, np.std(scores[5])*100))
	 
		
		
		i+=1
		los.append(scores[0]*100)
		acc.append(scores[1]*100)
		sen.append(scores[2]*100)
		spc.append(scores[3]*100)
		pre.append(scores[4]*100)
		f1s.append(scores[5]*100)

	save_results(acc, sen, spc, pre, f1s)

	print('################################# 10-Fold Cross Validation score #################################')
	print('los: %.2f%% (+/- %.2f%%)\nacc: %.2f%% (+/- %.2f%%)\nsen: %.2f%% (+/- %.2f%%)\n'
		  'spc: %.2f%% (+/- %.2f%%)\npre: %.2f%% (+/- %.2f%%)\nf1s: %.2f%% (+/- %.2f%%)' % 
		 ( np.mean(los), np.std(los), np.mean(acc), np.std(acc), np.mean(sen), np.std(sen),
		   np.mean(spc), np.std(spc), np.mean(pre), np.std(pre), np.mean(f1s), np.std(f1s)))


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