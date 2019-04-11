import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn import metrics


dataset = pd.read_csv("dataset.csv"
					,header=None)

print(dataset.head())

plt.scatter(dataset[0],dataset[1])
plt.savefig("scatter.png")

for i in range(5):
	n = i + 2
	# print(n)
	kmeans_predictions = KMeans(n_clusters=n).fit_predict(dataset)
	plt.scatter(dataset[0],dataset[1], c=kmeans_predictions)
	plt.savefig("scatter_kmean" + str(n) + ".png")
	print("kmean "  + str(n) + " clusters")
	print(metrics.silhouette_score(dataset, kmeans_predictions))

for i in range(5):
	n = i + 2
	# print(n)
	gaussian_predictions = GaussianMixture(n_components=n).fit(dataset).predict(dataset)
	plt.scatter(dataset[0],dataset[1], c=gaussian_predictions)
	plt.savefig("scatter_guassian" + str(n) + ".png")
	print("gaussian "  + str(n) + " components")
	print(metrics.silhouette_score(dataset, gaussian_predictions))

