import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 

BASE_URL = "5000000_Game_Data"
TRAIN_CSV = f"{BASE_URL}_Train-True.csv"
TEST_CSV = f"{BASE_URL}_Train-False.csv"
TRAIN_AVG_CSV = f"{BASE_URL}_Avg_Train-True.csv"
TEST_AVG_CSV = f"{BASE_URL}_Avg_Train-False.csv"
TRICK_CSV = "100000_Game_Data_Train-true.csv"


def plot_average_red_flags(train):
	data = pd.read_csv(TRAIN_AVG_CSV if train else TEST_AVG_CSV)
	rf_agent = list(map(float,(data["avg_rf_agent"][0][1:-1]).split(" ")))
	rf_random = list(map(float,(data["avg_rf_random"][0][1:-1]).split(" ")))
	labels = ['Flag 1', 'Flag 2', 'Flag 3']
	
	x_pos = np.array([i for i, _ in enumerate(labels)])
	width = 0.35

	ax = plt.subplot(111)
	ax.bar(x_pos, rf_agent, color='r', width=width, label="Agent")
	ax.bar(x_pos+width, rf_random, color='b', width=width, label="Random")

	plt.title(f'Average Red Flag Counts (in {"Training" if train else "Testing"})')
	plt.ylabel('Red Flag Ratio')
	plt.xticks(x_pos+width/2, labels)
	plt.legend(loc='best')
	plt.show()


FLAGS = {
	"a": "Plays highest card even though it will lose",
	"b": "Plays winning card if possible when last to play",
	"c": "Plays low card when partner plays winning card",
}
def plot_red_flags_over_time(flag, train):
	data = pd.read_csv(TRAIN_CSV if train else TEST_CSV)
	rfp_agent = data[f"agent_rfv_{flag}"] / data[f"agent_rftc_{flag}"]
	rfp_random = data[f"random_rfv_{flag}"] / data[f"random_rftc_{flag}"]

	xdata = list(range(len(data)))
	plt.plot(xdata, rfp_agent, "r.", alpha=0.8, markersize=0.5, label="Agent")
	plt.plot(xdata, rfp_random, "b.", alpha=0.8, markersize=0.5, label="Random")

	xdata = np.array(xdata).reshape(-1, 1)
	poly = PolynomialFeatures(degree = 4) 
	X_poly = poly.fit_transform(xdata)

	for rfp, color in zip([rfp_agent, rfp_random], ['red', 'blue']):
		poly.fit(X_poly, rfp)
		lin2 = LinearRegression() 
		lin2.fit(X_poly, rfp)
		plt.plot(xdata, lin2.predict(poly.fit_transform(xdata)), color=color)

	plt.suptitle(f'Red Flag Ratio Over Time (in {"Training" if train else "Testing"})')
	plt.title(FLAGS[flag], fontsize=10)
	plt.xlabel("Game Number (1000s)")
	plt.ylabel("Red Flag Ratio")
	plt.legend(loc='best', markerscale=12)
	plt.show()


def plot_moving_average_wins():
	df = pd.read_csv(TRICK_CSV)
	averages = df["diff_wins"].rolling(window=100).mean()
	cum_averages = df["diff_wins"].expanding(min_periods=10).mean()

	plt.plot(averages, ".", markersize=0.5, label="Moving Average")
	plt.plot(cum_averages, label="Cumulative Average")

	plt.title("Number of Rounds Won by Agent Compared to Random")
	plt.xlabel("Training Batch (10 games each)")
	plt.ylabel("Delta in Rounds Won")
	plt.legend(loc='best', markerscale=12)
	plt.show()


def main():
	plot_average_red_flags(train=False)
	# plot_red_flags_over_time(flag="a", train=True)
	# plot_red_flags_over_time(flag="b", train=True)
	# plot_red_flags_over_time(flag="c", train=True)
	# plot_moving_average_wins()


if __name__ == "__main__":
	main()