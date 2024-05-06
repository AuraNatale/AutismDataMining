
'''
We decide that to fill the missing values of the test subministred
it should be good to rely on the standard score achieved by the mean
of the population (if the statistics are available in the literature),
otherwise we will use the mean extracted from our dataset.
 '''

def test_score_fill (feature_name, feature_data):
    # We create a dictionary to store the literature mean scores
    literature_scores = {
    "FIQ": 97.34, #retrieved from 
    "model": "Mustang",
    "year": 1964
    }