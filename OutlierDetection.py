import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")


# PCA and KMeans method for outlier detection
# PCA and K-NN method for outlier detection

def getOutlier(method):
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_regular_season_career.txt", 8, 23,
	                "NBA Regular Seasons in general", method)
	print("===================================================================")
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_regular_season.txt", 6, 23,
	                "NBA Regular Seasons", method)
	print("===================================================================")
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_playoffs.txt", 8, 23, "NBA Playoffs",
	                method)
	print("===================================================================")
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_playoffs_career.txt", 8, 21,
	                "NBA Playoffs in General", method)
	print("===================================================================")

	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_allstar.txt", 8, 23, "NBA All Star",
	                method)


def readPlayerFiles(filename, start, end, name, method):
	df = pd.read_csv(filename, header=0).fillna(0)

	if "year" in df.columns:
		df1 = df.sort_values(by='year')
		print(df1)
		df1 = df1.reset_index(drop=True)
		print(df1)

		dataFrameArr = []
		years = []
		for year in df1["year"]:
			if (year not in years):
				years.append(year)

		for year in years:
			newDataFrame = df1.query("year==" + str(year))
			dataFrameArr.append(newDataFrame)

		for i in range(len(dataFrameArr)):
			newOutlier = dataFrameArr[i]
			if method == "KNN":
				outlier_KNN(newOutlier, years[i], start, end, name)
			else:
				outlier_KMeans(newOutlier, years[i], start, end, name)

	else:
		if method == "KNN":
			outlier_KNN(df, -1, start, end, name)
		else:
			outlier_KMeans(df, -1, start, end, name)


def perform_PCA(df):
	x = StandardScaler().fit_transform(df)
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data=principalComponents
	                           , columns=['principal component 1', 'principal component 2'])
	return principalDf


def get_Dist(x1, y1, x2, y2):
	diffX = x2 - x1
	if diffX > 0:
		return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
	else:
		return 0


def outlier_KMeans(df, year, start, end, name):
	df1 = df.iloc[:, start:end]

	principalDF = perform_PCA(df1)

	model = KMeans(n_clusters=1, random_state=0)
	identified_clusters = model.fit(principalDF[['principal component 1', 'principal component 2']])
	clusterCenter = model.cluster_centers_[0]
	x1 = clusterCenter[0]
	y1 = clusterCenter[1]

	# centeroids = model.cluster_centers_

	identified_clusters1 = identified_clusters.predict(principalDF[['principal component 1', 'principal component 2']])

	data_with_clusters = df.copy()

	data_with_clusters['Clusters'] = identified_clusters1
	data_with_clusters['PC1'] = 0.0
	data_with_clusters['PC2'] = 0.0
	startIndex = data_with_clusters.index[0]
	for i in range(len(data_with_clusters['firstname'])):
		data_with_clusters['PC1'][i + startIndex] = principalDF['principal component 1'][i]
		data_with_clusters['PC2'][i + startIndex] = principalDF['principal component 2'][i]

	# print(data_with_clusters)

	data_with_clusters['distFromCentre'] = 0.0
	for i in range(len(data_with_clusters['PC1'])):
		data_with_clusters['distFromCentre'][i + startIndex] = get_Dist(x1, y1,
		                                                                data_with_clusters["PC1"][i + startIndex],
		                                                                data_with_clusters['PC2'][i + startIndex])

	thresh = max(data_with_clusters['PC1']) - 2
	df = data_with_clusters.query("distFromCentre > " + str(thresh))

	if (year < 0):
		print('The Outstanding players for ' + name + ' are : ')
	else:
		print('The Outstanding players for ' + name + ' in the year ' + str(year) + ' are : ')
	print('================================================================================')
	print(df.index)
	for i in df.index:
		print(df['firstname'][i] + ' ' + df['lastname'][i] + ' ' + str(
			df['PC1'][i]) + '/' + str(df['PC2'][i]))

	print("===============================================================================")

	plt.scatter(principalDF['principal component 1'], principalDF['principal component 2'], color='r')
	plt.scatter(df['PC1'], df['PC2'],
	            color='b')

	plt.show()

	# TODO - split dataset into different positions
	# apply top 2 to other kNN and PCF method


def outlier_KNN(df1, year, start, end, name):
	df = df1.iloc[:, start:end].fillna(0)

	principalDf = self.perform_PCA(df)
	# create arrays
	X = principalDf.values

	nbrs = NearestNeighbors(n_neighbors=3)  # fit model
	nbrs.fit(X)
	# distances and indexes of k-neaighbors from model outputs
	distances, indexes = nbrs.kneighbors(X)  # plot mean of k-distances of each observation
	q1 = 0
	q3 = 0
	newDistances = []
	for d in distances:
		# print(d)
		newDistances.append(d.mean())

	q3, q1 = np.percentile(newDistances, [75, 25])
	iqr = q3 - q1
	lower_range = q1 - 1.5 * iqr
	upper_range = q3 + 1.5 * iqr

	player_playoffs_outlier_index = np.where(distances.mean(axis=1) > upper_range)

	startIndex = self.df.index[0]
	updatedOutlierIndices = []
	for i in range(len(player_playoffs_outlier_index[0])):
		updatedOutlierIndices.append(player_playoffs_outlier_index[0][i] + startIndex)

	outlier_values = principalDf.iloc[player_playoffs_outlier_index]
	Y = self.df.iloc[player_playoffs_outlier_index]
	# print(self.df)

	data_with_clusters = Y.copy()

	print(principalDf)
	data_with_clusters['PrincipalComponent1'] = 0.0
	data_with_clusters['PrincipalComponent2'] = 0.0

	for i in range(len(updatedOutlierIndices)):
		data_with_clusters['PrincipalComponent1'][updatedOutlierIndices[i]] = principalDf['principal component 1'][
			player_playoffs_outlier_index[0][i]]
		data_with_clusters['PrincipalComponent2'][updatedOutlierIndices[i]] = principalDf['principal component 2'][
			player_playoffs_outlier_index[0][i]]

	print(data_with_clusters)

	indexArray = []
	count = 0

	for index in updatedOutlierIndices:
		for out in outlier_values["principal component 1"]:
			if (data_with_clusters['PrincipalComponent1'][index] == out):
				indexArray.append(index)
		count = count + 1

	indexArray2 = []
	count = 0

	for index in updatedOutlierIndices:
		for out in outlier_values["principal component 2"]:
			if (data_with_clusters['PrincipalComponent2'][index] == out):
				indexArray2.append(index)
		count = count + 1

	finalIndexArray = []

	for m in indexArray:
		for n in indexArray2:
			if (m == n):
				finalIndexArray.append(m)

	if (year < 0):
		print('The Outstanding players for ' + name + ' are : ')
	else:
		print('The Outstanding players for ' + name + ' in the year ' + str(year) + ' are : ')
	print('================================================================================')
	for i in finalIndexArray:
		print(data_with_clusters['firstname'][i] + ' ' + data_with_clusters['lastname'][i] + ' ' + str(
			data_with_clusters['PrincipalComponent1'][i]) + '/' + str(data_with_clusters['PrincipalComponent2'][i]))

	print("===============================================================================")

	# if(data[data_with_clusters.columns.get_loc('PrincipalComponent1')] == out[outlier_values.columns.get_loc('principal component 1')] and data[data_with_clusters.columns.get_loc('PrincipalComponent2')] == out[outlier_values.columns.get_loc('principal component 2')] ):
	# print(data[1] + ' ' + data[2])

	# plot outlier values

	plt.scatter(principalDf["principal component 1"], principalDf["principal component 2"], color="r")
	plt.scatter(outlier_values["principal component 1"], outlier_values["principal component 2"], color="b")
	# print(data_with_clusters)
	plt.show()



if __name__ == "__main__":
    method = input("Enter the method to find outliers in the NBA (Enter KNN or KMeans):")
    print("OUTLIERS FOR NBA USING PCA and "+method)
    getOutlier(method)