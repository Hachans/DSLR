import pandas as pd
import matplotlib.pyplot as plt

def main():
	data = pd.read_csv("data/dataset_train.csv")
	slytherin = data[data["Hogwarts House"] == "Slytherin"]
	ravenclaw = data[data["Hogwarts House"] == "Ravenclaw"]
	gryffindor = data[data["Hogwarts House"] == "Gryffindor"]
	hufflepuff = data[data["Hogwarts House"] == "Hufflepuff"]

	houses = [slytherin, ravenclaw, gryffindor, hufflepuff]
	colors = ["green", "blue", "red", "yellow"]
	features = data.iloc[:, 6:]

	fig, axes = plt.subplots(ncols=13, nrows=13, figsize=(5, 5))
	plt.legend(["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"])
	fig.set_figwidth(13)
	fig.set_figheight(13)
	j = 0

	for column1 in features:
		k = 0
		for column2 in features:
			for i in range(len(houses)):
				if (column1 != column2):
					axes[j, k].scatter(y=houses[i][column1], x=houses[i][column2], color=colors[i], alpha=0.4)
				else:
					axes[j, k].hist(houses[i][column1].dropna(), color=colors[i], alpha=0.4)
				if (k == 0):
					axes[j, k].set_ylabel(column1)
				if (j == 12):
					axes[j, k].set_xlabel(column2)
				axes[j,k].set_xticks([])
				axes[j,k].set_yticks([])
			k += 1
		j += 1

	fig.tight_layout()

	plt.show()


if __name__ == "__main__":
	main()

