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
    if diffX >= 0:
        return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
    else:
        return 0


def outlierDetect(df, year, start, end, name):
    df1 = df.iloc[:, start:end]
    #print(df1)

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

    thresh1 = max(data_with_clusters['PC1'])-0.5
    thresh2  =  max(data_with_clusters['PC2'])-0.5
    df = data_with_clusters.query("PC1 > " + str(thresh1)+" or PC2 >"+str(thresh2))


    if (year < 0):
        print('The Outstanding players for ' + name + ' are : ')
    else:
        print('The Outstanding players for ' + name + ' in the year ' + str(year) + ' are : ')
    print('================================================================================')
    for i in df.index:
        print(df['firstname'][i] + ' ' + df['lastname'][i])
        #+ ' ' + str(df['PC1'][i]) + '/' + str(df['PC2'][i])

    print("================================================================================")

    plt.scatter(principalDF['principal component 1'], principalDF['principal component 2'], color='r')
    plt.scatter(df['PC1'], df['PC2'],
                color='b')

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
    readPlayerFiles("player_regular_season_career.txt", 8, 23,
                    "NBA Regular Seasons in general")

    print("===================================================================")
    readPlayerFiles("player_regular_season.txt", 6, 23,
                    "NBA Regular Seasons")
    print("===================================================================")
    readPlayerFiles("player_playoffs.txt", 8, 23, "NBA Playoffs")
    print("===================================================================")

    readPlayerFiles("player_playoffs_career.txt", 8, 21,
                    "NBA Playoffs in General")
    print("===================================================================")

    readPlayerFiles("player_allstar.txt", 8, 23, "NBA All Star")

