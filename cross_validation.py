from random import randrange
import numpy as np

def crossvalidation_split(data, folds):
	dataset_split = []
	dataset_copy = list(data)
	fold_size = int(len(data) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


# folds = crossvalidation_split(Dataset, 10)
# items = np.array([folds[i][0:len(folds[i])-1] for i in range(len(folds))])
# targets = np.array([folds[i][len(folds[i])-1] for i in range(len(folds))])
# print(len(items[0][0]))
# print(targets)
# print(items)
# print(len(folds))