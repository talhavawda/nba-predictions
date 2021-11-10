import pandas as pd
from joblib import load
import GameOutcomeTraining

"""
	Prediction the winner of an NBA game using our best trained model given the feature vectors (of the 31 features) of 2 teams

	These feature vectors are stored in a csv file, with each feature vector being stored on a line, and its feature values being
	separated by commas. The first line represents the feature vector for Team 1 and the second line represents the feature
	vector for Team 2
"""

def main():
	print("NBA GAME OUTCOME PREDICTION")
	print("============================")

	model = load("MLPRegressor.joblib")

	filePath = input("\nEnter the name of the textfile (located in the current directory) containing the feature vectors of the 2 teams: ")
	# GameOutcomeTest1.csv

	try:
		featureVectorsFile = open(filePath, "r")
	except FileNotFoundError:
		print("Invalid file name entered. This program shall terminate.")
		quit()

	features = ["o_fgm", "o_fga", "o_ftm", "o_fta", "o_oreb", "o_dreb", "o_reb", "o_asts", "o_pf", "o_stl", "o_to", "o_blk", "o_3pm", "o_3pa", "o_pts", "d_fgm", "d_fga", "d_ftm", "d_fta", "d_oreb", "d_dreb", "d_reb", "d_asts", "d_pf", "d_stl", "d_to", "d_blk", "d_3pm", "d_3pa", "d_pts", "pace"]
	featureVectorsDF = pd.read_csv(filePath, names=features)
	# print(featureVectorsDF)
	print()
	print("Team 1's feature vector:")
	print(featureVectorsDF.iloc[0].values.tolist())

	print()
	print("Team 2's feature vector:")
	print(featureVectorsDF.iloc[1].values.tolist())

	# Do the win probability predictions
	winProbabilityPredictions = model.predict(featureVectorsDF)
	winProbabilityTeam1 = winProbabilityPredictions[0]
	winProbabilityTeam2 = winProbabilityPredictions[1]

	"""
		Since the win probability prediction for each team is for that team in general only, normalise the probabilities
		such that its relative to the other team and the sum of the 2 win probabilities add up till 1 
	"""
	normalisedWinProbabilityTeam1 = winProbabilityTeam1 / (winProbabilityTeam1 + winProbabilityTeam2)
	normalisedWinProbabilityTeam2 = winProbabilityTeam2 / (winProbabilityTeam1 + winProbabilityTeam2)

	print()
	print("The probability of Team 1 beating Team 2 is: ", normalisedWinProbabilityTeam1)
	print("The probability of Team 2 beating Team 1 is: ", normalisedWinProbabilityTeam2)


	if (normalisedWinProbabilityTeam1 > normalisedWinProbabilityTeam2):
		winner = "Team 1"
	else:
		winner = "Team 2"

	print("Thus the winner between Team 1 and Team 2 is: ", winner)

if __name__ == "__main__":
	main()
