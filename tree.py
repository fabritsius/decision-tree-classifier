import numpy as np
import pandas as pd

class DecisionTreeClassifier():
	""" This model, if given some labeled data, can be trained to predict labels of new data. 
		Functions:
			fit(X, Y) – train;
			predict(X) – use.
	"""
	def __init__(self, tree=None):
		self.tree = tree

	def fit(self, X, Y):
		""" Train the model.
			Parameters:
				X – data for training;
				Y – corresponding data labels;
			Returns:
				self – this instance of the class.
		"""
		self.tree = []
		data = pd.DataFrame(X, Y)

		def divide(data):
			""" Split the data in the best spot. """
			# finish this branch if it is pure
			if len(set(data.index)) == 1:
				return data.index[0]
			# find the best split otherwise
			data_length = len(data)
			SPLITS = [0.25, 0.5, 0.75]
			lowest_G = 1
			best_buckets = None
			for col in data.columns:
				sorted_col = data[col].sort_values()
				for split in SPLITS:
					this_bucket_end = int(data_length*split)
					next_bucket_start = int(data_length*split) + 1
					
					this_bucket = sorted_col[:next_bucket_start]
					next_bucket = sorted_col[next_bucket_start:]

					this_len = len(this_bucket)
					next_len = len(next_bucket)

					this_uniq, this_occurs = np.unique(this_bucket.index, return_counts=True)
					next_uniq, next_occurs = np.unique(next_bucket.index, return_counts=True)

					# G = 1 – ( P(class1)^2 + P(class2)^2 + … + P(classN)^2)
					G_this = 1 - sum([c/this_len*c/this_len for c in this_occurs])
					G_next = 1 - sum([c/next_len*c/next_len for c in next_occurs])

					G_sum = G_this + G_next
					if G_sum < lowest_G:
						lowest_G = G_sum
						best_buckets = (col, this_bucket, next_bucket)
			
			b1 = best_buckets[1].ix[-1]
			b2 = best_buckets[2].ix[0]
			best_split = (best_buckets[0], (b1 + b2) / 2)
			
			sorted_data = data.sort_values(best_buckets[0])
			branch_1 = sorted_data[len(best_buckets[1]):]
			branch_2 = sorted_data[:len(best_buckets[1])]
			return best_split, branch_1, branch_2

		def grow_tree(tree, path, value):
			""" Add 'value' to a 'tree' object, while kipping in mind given 'path'. """
			if path:
				current = tree
				for branch in path[:-1]:
					current = current[branch]
				current[path[-1]] = value
			else:
				tree += value

		unfinished_branches = [([], data)]
		while unfinished_branches:
			this_branch = unfinished_branches.pop()
			next_branches = divide(this_branch[1])
			if isinstance(next_branches, tuple):
				# grow two new branches
				next_branch_value = [next_branches[0], [], []]
				grow_tree(self.tree, this_branch[0], next_branch_value)
				unfinished_branches.append((this_branch[0] + [1], next_branches[1]))
				unfinished_branches.append((this_branch[0] + [2], next_branches[2]))
			else:
				# create one leaf (finish this branch)
				grow_tree(self.tree, this_branch[0], next_branches)
		# return entire class		
		return self

	def predict(self, X):
		""" Use tree to predict Y (class) from X (input data). """
		predictions = []
		for x in X:
			current = self.tree
			while True:
				b = current[0]
				if isinstance(b, tuple):
					if x[b[0]] > b[1]:
						current = current[1]
					else:
						current = current[2]
				else:
					predictions.append(current)
					break
		# return list of predictions
		return predictions