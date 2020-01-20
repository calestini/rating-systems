import sys
import numpy as np

## muted in global import for production
# from sklearn.metrics import auc, roc_curve, confusion_matrix, classification_report

from scipy import interp
from itertools import cycle
import itertools

## muted in global import for production
# import matplotlib.pyplot as plt

"""
http://scikit-learn.org/stable/modules/model_evaluation.html
"""
#
# tie_target_names = ['Away Win', 'Home Win', 'Tie']
# nontie_target_names = ['Away Win', 'Home Win']
#
#
# def progress(count, total, status=''):
# 	"""
# 	Adapted from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
# 	"""
# 	bar_len = 60
# 	filled_len = int(round(bar_len * count / float(total)))
#
# 	percents = round(100.0 * count / float(total), 1)
# 	bar = '=' * filled_len + '-' * (bar_len - filled_len)
#
# 	sys.stdout.write('[{0}] {1}{2} ... {3}\r'.format(bar, percents, '%', status))
# 	sys.stdout.flush()
#
#
# def log_loss(s, p):
# 	"""
# 	Calculates the log loss given two np arrays: actual outcome and probability projection
# 	"""
# 	p = np.maximum(np.minimum(p, 1 - 10**-15), 10**-15)
# 	arr = -(s * np.log10(p) + (1 - s) * np.log10(1 - p))
#
# 	return np.mean(arr)
#
#
# def mlog_loss(true, pred):
# 	"""
# 	Calculate multi-class log loss
# 	"""
# 	l_loss = dict()
#
# 	target_names = tie_target_names
#
# 	for i, class_name in enumerate(target_names):
# 		l_loss[class_name] = -true[:, i]*np.log10(pred[:, i])
#
# 	l_loss_all = np.sum([l_loss[class_name] for class_name in target_names], axis =0)
#
# 	return np.mean(l_loss_all)
#
#
# def squared_error(s, p):
# 	"""
# 	Calculates the mean squared error given two np arrays: actual outcome and probability projection
# 	"""
# 	arr = np.power((s-p), 2)
# 	return np.mean(arr)
#
#
# def conf_matrix(s, p):
# 	"""
# 	Calculates the confusion matrix
# 	"""
# 	return confusion_matrix(s, p)
#
#
# def graph_confusion_matrix(cm):
# 	"""
# 	Graphs the confusion matrix
# 	"""
# 	n_classes = cm.shape[1]
#
# 	if n_classes == 3:
# 		target_names = tie_target_names
# 	else:
# 		target_names = nontie_target_names
#
# 	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# 	plt.title('Confusion Matrix')
# 	plt.colorbar()
# 	tick_marks = np.arange(len(target_names))
# 	plt.xticks(tick_marks, target_names, rotation=45)
# 	plt.yticks(tick_marks, target_names)
#
# 	fmt = 'd'
# 	thresh = cm.max() / 2.
# 	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# 		plt.text(j, i, format(cm[i, j], fmt),
# 				 horizontalalignment="center",
# 				 color="white" if cm[i, j] > thresh else "black")
#
# 	plt.tight_layout()
# 	plt.ylabel('True label')
# 	plt.xlabel('Predicted label')
# 	plt.show()
#
#
# def classif_report(s, p):
# 	"""
# 	Displays the classification report
# 	"""
# 	n_classes = np.unique(s).sum()
# 	if n_classes == 3:
# 		target_names = tie_target_names
# 	elif n_classes == 2:
# 		target_names = nontie_target_names
# 	else:
# 		target_names = 1
#
# 	return classification_report(s, p, target_names=target_names)
#
#
# def roc_auc_s(true, pred):
# 	"""
# 	Calculates the ROC
# 	"""
# 	n_classes = true.shape[1]
#
# 	if n_classes == 3:
# 		target_names = tie_target_names
#
# 		fpr = dict()
# 		tpr = dict()
# 		roc_auc = dict()
#
# 		for i, class_name in enumerate(target_names):
# 			fpr[class_name], tpr[class_name], _ = roc_curve(true[:, i], pred[:, i])
# 			roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
#
# 		# Compute micro-average ROC curve and ROC area
# 		fpr["micro"], tpr["micro"], _ = roc_curve(true.ravel(), pred.ravel())
# 		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# 		# First aggregate all false positive rates
# 		all_fpr = np.unique(np.concatenate([fpr[class_name] for class_name in target_names]))
#
# 		# Then interpolate all ROC curves at this points
# 		mean_tpr = np.zeros_like(all_fpr)
#
# 		for class_name in target_names:
# 			mean_tpr += interp(all_fpr, fpr[class_name], tpr[class_name])
#
# 		# Finally average it and compute AUC
# 		mean_tpr /= n_classes
#
# 		fpr["macro"] = all_fpr
# 		tpr["macro"] = mean_tpr
# 		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# 	else:
# 		fpr, tpr, _ = roc_curve(true, pred)
# 		roc_auc = auc(fpr, tpr)
#
# 	return roc_auc, fpr, tpr
#
#
# def graph_roc_auc(roc_auc, fpr, tpr, n_classes):
# 	"""
# 	Graphs ROC curves
# 	"""
# 	if n_classes == 3:
# 		target_names = tie_target_names
# 		lw = 2
# 		plt.figure()
# 		plt.plot(fpr["micro"], tpr["micro"],
# 				 label='micro-average ROC curve (area = {0:0.2f})'
# 					   ''.format(roc_auc["micro"]),
# 				 color='deeppink', linestyle=':', linewidth=4)
#
# 		plt.plot(fpr["macro"], tpr["macro"],
# 				 label='macro-average ROC curve (area = {0:0.2f})'
# 					   ''.format(roc_auc["macro"]),
# 				 color='navy', linestyle=':', linewidth=4)
#
# 		colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# 		for color, class_name in zip(colors, target_names):
# 			plt.plot(fpr[class_name], tpr[class_name], color=color, lw=lw,
# 					 label='ROC curve of class {0} (area = {1:0.2f})'
# 					 ''.format(class_name, roc_auc[class_name]))
# 	else:
# 		plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'
# 				 ''.format(roc_auc), color='aqua', linestyle=':', linewidth=2)
#
# 	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# 	plt.xlim([0.0, 1.0])
# 	plt.ylim([0.0, 1.05])
# 	plt.xlabel('False Positive Rate')
# 	plt.ylabel('True Positive Rate')
# 	plt.title('Receiver Operating Characteristic')
# 	plt.legend(loc="lower right")
# 	plt.show()


def _get_name(array, optional, needed=['']):
	"""
	Loop through array until name has part of <needed[i]> and of <optional>
	"""
	for has_t in optional:
		for i in array:
			if has_t in i:
				for need_it in needed:
					if need_it in i:
						array.remove(i)
						return i, array

	raise ValueError(f'Could not find a match for {optional}')


def get_names(df, simplified=False):
	"""
	Use a string match to find out which fields are which, if not specified.
	"""
	home_prefixes = ['home','local']
	vis_prefixes = ['vis','away']
	date_prefixes = ['date','hour','start','order','sequence']

	cols = df.columns.tolist()

	home_col, cols = _get_name(cols, ['home','local'], ['teamid','team_id','id'])
	vis_col, cols = _get_name(cols, ['vis','away'], ['teamid','team_id','id'])

	if simplified:
		return home_col, vis_col

	home_score, cols =  _get_name(cols, ['home','local','h','l'], ['score','points','pts','goals'])
	vis_score, cols =  _get_name(cols, ['vis','away','v','a'], ['score','points','pts','goals'])
	season, cols = _get_name(cols, ['season','year','league'])
	start_datetime, cols = _get_name(cols, ['date','hour','start','order','sequence'])

	return home_col, vis_col, home_score, vis_score, season, start_datetime
