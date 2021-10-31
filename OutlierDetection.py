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
# TOADD PCA and K-NN method for outlier detection

def perform_PCA(df):
	x = StandardScaler().fit_transform(df)
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data=principalComponents
	                           , columns=['principal component 1', 'principal component 2'])
	return principalDf


def get_Dist(x1, y1, x2, y2):
	diffX = x2 - x1
	diffY = y2 - y1
	gradient = diffY / diffX
	if gradient > 0 and diffY > 0 and diffX > 0:
		return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
	else:
		return 0


def outlierDetect(df, year, start, end, name):
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


'''def getPositions(df):
   playerDF = pd.read_csv("players.txt",header=0)
   merged = pd.merge(df,playerDF,on="ilkid")'''


def readPlayerFiles(filename, start, end, name):
	df1 = pd.read_csv(filename, header=0).fillna(0)

	if "year" in df1.columns:
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
			outlierDetect(newOutlier, years[i], start, end, name)

	else:
		outlierDetect(df1, -1, start, end, name)


if __name__ == "__main__":
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_regular_season_career.txt", 8, 23,
	                "Regular Seasons in general")
	print("===================================================================")
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_regular_season.txt", 6, 23,
	                "Regular Seasons")
	'''print("===================================================================")
	readPlayerFiles("player_playoffs.txt")
	print("===================================================================")
	readPlayerFiles("player_playoffs_career.txt")
	print("===================================================================")
	readPlayerFiles("player_allstar.txt")'''

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
# TOADD PCA and K-NN method for outlier detection

def perform_PCA(df):
	x = StandardScaler().fit_transform(df)
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data=principalComponents
	                           , columns=['principal component 1', 'principal component 2'])
	return principalDf


def get_Dist(x1, y1, x2, y2):
	diffX = x2 - x1
	diffY = y2 - y1
	gradient = diffY / diffX
	if gradient > 0 and diffY > 0 and diffX > 0:
		return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
	else:
		return 0


def outlierDetect(df, year, start, end, name):
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


'''def getPositions(df):
   playerDF = pd.read_csv("players.txt",header=0)
   merged = pd.merge(df,playerDF,on="ilkid")'''


def readPlayerFiles(filename, start, end, name):
	df1 = pd.read_csv(filename, header=0).fillna(0)

	if "year" in df1.columns:
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
			outlierDetect(newOutlier, years[i], start, end, name)

	else:
		outlierDetect(df1, -1, start, end, name)


if __name__ == "__main__":
	# Todo - change directory of files (originally it was on Ricarod's G Drive folder)
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_regular_season_career.txt", 8, 23,
	                "Regular Seasons in general")
	print("===================================================================")
	readPlayerFiles("drive/MyDrive/Comp721 Project/databasebasketball/player_regular_season.txt", 6, 23,
	                "Regular Seasons")
	'''print("===================================================================")
	readPlayerFiles("player_playoffs.txt")
	print("===================================================================")
	readPlayerFiles("player_playoffs_career.txt")
	print("===================================================================")
	readPlayerFiles("player_allstar.txt")'''

