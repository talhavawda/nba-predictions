import pandas as pd
import GameOutcomeTraining

"""
	Prediction the winner of an NBA game using our best trained model given the feature vectors (of the 31 features) of 2 teams

	These feature vectors are stored in a csv file, with each feature vector being stored on a line, and its feature values being
	separated by commas
"""

def main():
	print("NBA GAME OUTCOME PREDICTION")
	print("============================")

	model = load("mlpRegressor.joblib")

	filePath = input("\nEnter the name of the textfile (located in the current directory) containing the feature vectors of the 2 teams: ")

	try:
		featureVectorsFile = open(filePath, "r")
	except FileNotFoundError:
		print("Invalid file name entered. This program shall terminate.")
		quit()

	features = ["o_fgm","o_fga","o_ftm","o_fta","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_3pm","o_3pa","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_3pm","d_3pa","d_pts","pace"]
	featureVectorsDF = pd.read_csv(filePath, names=features)
	print(featureVectorsDF)

if __name__ == "__main__":
	main()
