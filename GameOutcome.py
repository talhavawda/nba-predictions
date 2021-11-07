import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics, preprocessing

"""
	Game Prediction: Predicting the outcome of a game, given 2 teams.
		
		Since we do not have any (head-to-head) data (from the provided dataset) that shows the results of actual games 
		(between two teams), this is an Unsupervised Machine Learning task.
		
		However, we do have the number of wins and losses for each team per season, so we can use those numbers to obtain
		the win rate of the team for that season, which we can use to represent the probability of a team winning.	
		
		
		Using data from 1976 onwards as this is when the ABA merged into the NBA and most stats are specified in the team_season.txt
		dataset file (which represents regular season team stats). Number of games each team plays in a regular season 
		(from 1976 onwards) is 82. The year specified indicates the year the season started. E.g. the 2004-2005 season will have year=2004
		3pa stat for both offensive (o) and defensive (d) is only specified from 1999 onwards.
		
		Offensive stats are stats for this team, and defensive stats are stats against this team (i.e. stats conceded by this team)
		
		Features (the stats) using: 
			31 features from team_season.txt - 15 offensive (o) stats, 15 defensive (d) stats, and the pace stat
			
		We can use the win rate of the team-season entry as the label if we're going the Supervised Learning route
		
		Possible method:
		A test data sample instance is a feature-vector of the 31-instances and the prediction (if using SL) can be 
		the predicted win rate, and we take two test samples (using two teams) and the team with the higher predicted win 
		rate should win the match (we can maybe use one hot encoding for the NN)
"""


#3. Linear Regression?
#4  Measure the probability of each team winning

# ANNs
# use regression wit win rate as the labels?
# use recurrent NN to favour more recent seasons?

# Training can be on all the data in the dataset file with the labels being the win rate (a probability value), and test data can be a team with
# all the features specified, and goal is to output the probability of winning


"""
	Data Acquisition and Pre-processing
"""

# Read the team_season.txt dataset file into a DataFrame
teamSeasonsFile = "data/team_season.txt"
teamSeasonDF = pd.read_csv(teamSeasonsFile, header=0)  # First line of data is taken as the column headings
#print(teamSeasonDF)

# Remove entries where year < 1976 (Since we're only working with teams' season data from 1976 onwards
teamSeasonDF.drop(teamSeasonDF[teamSeasonDF.year < 1976].index, inplace=True)
teamSeasonDF.reset_index(inplace=True, drop=True)  #drop=True - don't add old indexes as a column
#print(teamSeasonDF)


"""
	Get the Data Frame with the win rate for each team-season entry - this will be the labels

	The pandas function apply(), which is called on a DataFrame, takes in a function as a parameter and iterates
	through each row of the DataFrame and applies the function to that row of the DataFrame (i.e. it sets the value of the 
	row's cell(s) to the value returned by the function for that row)

	So we shall pass in our get_win_rate() function to teamSeasonDF.apply() to get the win rate for each team-season entry
	and store the win rates in a corresponding DataFrame
"""

def get_win_rate(teamSeasonRow):
	"""
		Calculate and return the win rate of a team-season entry.
		The win rate is the proportion of wins to the number of matches played.
		Since there is no specific matches played attribute, matches played = number of wins + number of losses

		:param teamSeasonRow: A row (team-season entry) from the teamSeasonDF
		:return: the win rate for this team-season entry
	"""

	return teamSeasonRow["won"] / (teamSeasonRow["won"] + teamSeasonRow["lost"])


labelsDF = pd.DataFrame(columns=["win_rate"])
labelsDF["win_rate"] = teamSeasonDF.apply(get_win_rate, axis=1)
#print(labelsDF)


"""
	Since the 3pm stats are not provided for 1976, 1977, and 1978 (the values are 0 in the dataset),
	we're going to calculate the average 3pm stats value from the entries from 1979 onwards, and for each team-season entry 
	from 1976-1978, we're going to calculate its win-rate's percentage difference from the average win rate and apply it to 
	the average 3pm stats value to get the 3pm stats value for this team-season.
	The average win rate will be 0.5 since for each game a team won, the other team lost (We have also verified this value below)
"""

averageWinRate = teamSeasonDF["won"].sum() / (teamSeasonDF["won"].sum() + teamSeasonDF["lost"].sum())  # is equal to 0.5

teamSeason1979DF = teamSeasonDF[teamSeasonDF.year >= 1979]

o_3pmDF = teamSeason1979DF["o_3pm"]
o_3pmAverage = o_3pmDF.sum() / len(teamSeason1979DF.index)
#print("o_3pmAverage:", o_3pmAverage)

d_3pmDF = teamSeason1979DF["d_3pm"]
d_3pmAverage = d_3pmDF.sum() / len(teamSeason1979DF.index)
#print("d_3pmAverage:", d_3pmAverage)


def get_o_3pm_value(teamSeasonRow):
	"""
		Calculate and return the approximate o_3pm value of this team-season entry for when the o_3pm stat value was
		not provided (1976-1978), otherwise return the team-season entry's actual o_3pm value

		:param teamSeasonRow: A row (team-season entry) from the teamSeasonDF
		:return: the o_3pm value for this team-season entry
	"""

	teamSeasonWinRate = get_win_rate(teamSeasonRow)

	if teamSeasonRow["year"] < 1979:  # i.e. the year is 1976, 1977, or 1978
		# averageWinRate and o_3pmAverage is accessible here since this function is a sub-function of the main script
		o_3pmMultiplier = 1 + ((teamSeasonWinRate - averageWinRate) / averageWinRate)
		return int(round(o_3pmAverage * o_3pmMultiplier, ndigits=0))
	else:
		return teamSeasonRow["o_3pm"]


def get_d_3pm_value(teamSeasonRow):
	"""
		Calculate and return the approximate d_3pm value of this team-season entry for when the d_3pm stat value was
		not provided (1976-1978), otherwise return the team-season entry's actual d_3pm value

		:param teamSeasonRow: A row (team-season entry) from the teamSeasonDF
		:return: the d_3pm value for this team-season entry
	"""

	teamSeasonWinRate = get_win_rate(teamSeasonRow)

	if teamSeasonRow["year"] < 1979:  # i.e. the year is 1976, 1977, or 1978
		# averageWinRate and d_3pmAverage is accessible here since this function is a sub-function of the main script
		d_3pmMultiplier = 1 + ((teamSeasonWinRate - averageWinRate) / averageWinRate)
		return int(round(d_3pmAverage * d_3pmMultiplier, ndigits=0))
	else:
		return teamSeasonRow["d_3pm"]

teamSeasonDF["o_3pm"] = teamSeasonDF.apply(get_o_3pm_value, axis=1)
teamSeasonDF["d_3pm"] = teamSeasonDF.apply(get_d_3pm_value, axis=1)


"""
	Since the 3pa stats are only available from 1999 onwards, for the seasons from 1976 - 1998 (where 3pa is 0), 
	we're going to calculate the average multiplier (that links 3pm to 3pa) from the entries from 1999 onwards and 
	apply it to all the entries pre-1999 to get their 3pa stat (3pa = 3pm * multiplier) 
	[Alternatively we could not use the 3pa stat as a feature. Think about this.]
"""

teamSeason1999DF = teamSeasonDF[teamSeasonDF.year >= 1999]
#print(teamSeason1999DF)
o_3pmDF = teamSeason1999DF["o_3pm"]
o_3paDF = teamSeason1999DF["o_3pa"]
o_3paMultiplier = o_3paDF.sum() / o_3pmDF.sum()

d_3pmDF = teamSeason1999DF["d_3pm"]
d_3paDF = teamSeason1999DF["d_3pa"]
d_3paMultiplier = d_3paDF.sum() / d_3pmDF.sum()

#print(o_3pmDF.sum(), o_3paMultiplier)
#print(d_3pmDF.sum(), d_3paMultiplier)


def get_o_3pa_value(teamSeasonRow):
	"""
		Calculate and return the approximate o_3pa value of this team-season entry (based on the average provided o_3pa/o_3pm value)
		for when the o_3pa stat value was not provided (pre-1999), otherwise return the team-season entry's actual o_3pa value
		:param teamSeasonRow: A row (team-season entry) from the teamSeasonDF
		:return: the o_3pa value for this team-season entry
	"""
	# o_3paMultiplier - The average value of o_3pa/o_3pm for when the o_3pa stat value was provided | accessible here since this function is a sub-function of the main script
	if teamSeasonRow["year"] < 1999:
		return int(round(teamSeasonRow["o_3pm"] * o_3paMultiplier, ndigits=0))
	else:
		return teamSeasonRow["o_3pa"]


def get_d_3pa_value(teamSeasonRow):
	"""
		Calculate and return the approximate d_3pa value of this team-season entry (based on the average provided d_3pa/d_3pm value)
		for when the d_3pa stat value was not provided (pre-1999), otherwise return the team-season entry's actual d_3pa value
		:param teamSeasonRow: A row (team-season entry) from the teamSeasonDF
		:return: the d_3pa value for this team-season entry
	"""
	# d_3paMultiplier - The average value of d_3pa/d_3pm for when the d_3pa stat value was provided | accessible here since this function is a sub-function of the main script
	if teamSeasonRow["year"] < 1999:
		return int(round(teamSeasonRow["d_3pm"] * d_3paMultiplier, ndigits=0))
	else:
		return teamSeasonRow["d_3pa"]


teamSeasonDF["o_3pa"] = teamSeasonDF.apply(get_o_3pa_value, axis=1)
teamSeasonDF["d_3pa"] = teamSeasonDF.apply(get_d_3pa_value, axis=1)
#print(teamSeasonDF)


# Get a DataFrame containing the features only - remove "team", "year", "leag", "won", "lost" attributes
featureMatrixDF = teamSeasonDF.drop(["team", "year", "leag", "won", "lost"], axis=1)
print(featureMatrixDF)



# Do Data Normalisation (scaling the data)

# See https://scikit-learn.org/stable/modules/preprocessing.html
scaler = preprocessing.StandardScaler()


# TODO - (if splitting dataset into training and testing) only call fit_transform() on the Training Data only (and not the entire dataset, which also contains the test data)
# as we dont want the model to learn/include the the test data when calculating the mean/variance in fit(), but we'll apply them to the test data, so
# we'll therafter call transform() onto the test data
# See: https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
# See: https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models



# Get the normalised featureMatrixDF
normalisedFMDF = pd.DataFrame(scaler.fit_transform(featureMatrixDF), columns=featureMatrixDF.columns)  # Normalise feature matrix and convert back to a DataFrame
print(normalisedFMDF)
print(labelsDF)
print(normalisedFMDF.iloc[0])

# Do PCA?
# See what PCA is: https://builtin.com/data-science/step-step-explanation-principal-component-analysis
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html



"""
	Training
	- Using keras/tensorflow for the NN modelling
"""


# Training can be on all the data in the dataset file with the labels being the win rate (a probability value), and test data can be a team with
# all the features specified, and goal is to output the probability of winning



