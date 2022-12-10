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

	for i in range(len(houses)):
		plt.hist(houses[i]["Care of Magical Creatures"].dropna(), color=colors[i], alpha=0.4)
		plt.title("Care of Magical Creatures")
		plt.legend(["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"])
	plt.show()

if __name__ == "__main__":
	main()