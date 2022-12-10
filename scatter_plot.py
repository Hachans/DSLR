import pandas as pd
import matplotlib.pyplot as plt

def main():
	data = pd.read_csv("data/dataset_train.csv")
	plt.scatter(data["Astronomy"], data["Defense Against the Dark Arts"])
	plt.xlabel("Astronomy")
	plt.ylabel("Defense Against the Dark Arts")
	plt.show()

if __name__ == "__main__":
	main()