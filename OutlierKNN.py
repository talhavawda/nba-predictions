import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib
import matplotlib.pyplot as plt
import copy
from scipy.stats import iqr
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")



def perform_PCA(df):
    x = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    return principalDf


def outlierDetect(df1, year, start, end, name):
    df = df1.iloc[:, start:end].fillna(0)

    principalDf = perform_PCA(df)
    # create arrays
    X = principalDf.values

    nbrs = NearestNeighbors(n_neighbors=3)  # fit model
    nbrs.fit(X)
    # distances and indexes of k-neaighbors from model outputs
    distances, indexes = nbrs.kneighbors(X)  # plot mean of k-distances of each observation

    player_playoffs_outlier_index = np.where(distances.mean(axis=1) > 1)

    startIndex = df.index[0]
    updatedOutlierIndices = []
    for i in range(len(player_playoffs_outlier_index[0])):
        updatedOutlierIndices.append(player_playoffs_outlier_index[0][i] + startIndex)

    outlier_values = principalDf.iloc[player_playoffs_outlier_index]
    Y = df1.iloc[player_playoffs_outlier_index]


    data_with_clusters = Y.copy()


    data_with_clusters['PrincipalComponent1'] = 0.0
    data_with_clusters['PrincipalComponent2'] = 0.0
    # print(data_with_clusters)
    # print(data_with_clusters)
    for i in range(len(updatedOutlierIndices)):
        data_with_clusters['PrincipalComponent1'][updatedOutlierIndices[i]] = principalDf['principal component 1'][
            player_playoffs_outlier_index[0][i]]
        data_with_clusters['PrincipalComponent2'][updatedOutlierIndices[i]] = principalDf['principal component 2'][
            player_playoffs_outlier_index[0][i]]


    indexArray = []
    count = 0

    # print(player_playoffs_outlier_index[0])
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
        print(data_with_clusters['firstname'][i] + ' ' + data_with_clusters['lastname'][i])
        #+ ' ' + str(data_with_clusters['PrincipalComponent1'][i]) + '/' + str(data_with_clusters['PrincipalComponent2'][i])

    print("===============================================================================")



    # plot outlier values

    plt.scatter(principalDf["principal component 1"], principalDf["principal component 2"], color="r",)
    plt.scatter(outlier_values["principal component 1"], outlier_values["principal component 2"], color="b")
    # print(data_with_clusters)
    plt.show()


def readPlayerFiles(filename, start, end, name):
    df = pd.read_csv(filename, header=0).fillna(0)


    if "year" in df.columns:
        df1 = df.sort_values(by='year')

        df1 = df1.reset_index(drop=True)


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
            print()
            print()

    else:
        outlierDetect(df, -1, start, end, name)
        print()
        print()


if __name__ == "__main__":
    readPlayerFiles("data/player_regular_season_career.txt", 8, 23,
                    "NBA Regular Seasons in general")

    print("===================================================================")

    readPlayerFiles("data/player_playoffs_career.txt", 8, 21,
                    "NBA Playoffs in General")

    print("===================================================================")

    readPlayerFiles("data/player_regular_season.txt", 6, 23,
                    "NBA Regular Seasons")
    print("===================================================================")
    readPlayerFiles("data/player_playoffs.txt", 8, 23,
                    "NBA Playoffs")
    print("===================================================================")

    readPlayerFiles("data/player_allstar.txt", 8, 23,
                    "NBA All Star")
