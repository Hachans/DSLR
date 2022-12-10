import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import sys

def max(data):
	n = data[0]
	for i in range(len(data)):
		if (data[i] > n):
			n = data[i]
	return n

def min(data):
	n = data[0]
	for i in range(len(data)):
		if (data[i] < n):
			n = data[i]
	return n

def std(data, mean, count):
	variance = sum((x - mean) ** 2 for x in data) / (count - 1)
	v_std = np.sqrt(variance)
	return v_std

def describe(data, column):
	existing = pd.notnull(data)
	count = len(existing[existing == True])
	mean = np.sum(data[existing]) / count
	v_std = std(data[existing], mean, count)
	v_min = min(data)
	v_max = max(data)
	quarter = data[existing].sort_values().iloc[int(count * 0.25)]
	half = data[existing].sort_values().iloc[int(count * 0.5)]
	three_quarters = data[existing].sort_values().iloc[int(count * 0.75)]
	d = [count, mean, v_std, v_min, quarter, half, three_quarters, v_max]
	return d

def main():
	data = pd.read_csv(sys.argv[1])
	d = pd.DataFrame(index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"])

	for column in data:
		if (is_numeric_dtype(data[column])):
			d.insert(len(d.columns), column, describe(data[column], column))
	print(d)


if __name__ == "__main__":
	try:
		main()
	except:
		print("Something went wrong... Make sure you provided the right path to your dataset", file=sys.stderr)