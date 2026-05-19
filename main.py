import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# DATA LOADING 
df = pd.read_csv('C:\The Spot 2\Python\IPL Prediction Tool\IPL.csv')

# DATA CLEANING
matches = df.groupby('match_id').agg(
    date=('date', 'first'),
    team1=('batting_team', 'first'),
    team2=('bowling_team', 'first'),
    venue=('venue', 'first'),
    toss_winner=('toss_winner', 'first'),
    toss_decision=('toss_decision', 'first'),
    winner=('match_won_by', 'first'),
    season=('year', 'first')
).reset_index()

#same team/team rebrands
name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Kings XI Punjab': 'Punjab Kings'
}

matches['team1'] = matches['team1'].replace(name_map)
matches['team2'] = matches['team2'].replace(name_map)
matches['winner'] = matches['winner'].replace(name_map)
matches['toss_winner'] = matches['toss_winner'].replace(name_map)


# drop unknown/washed-off results
matches = matches[matches['winner'] != 'Unknown'].reset_index(drop=True)

#columns for toss and game winners 
matches["toss_winner_is_team1"] = np.where(matches["toss_winner"]==matches['team1'], 1,0)
matches["target"] = np.where(matches["winner"]==matches["team1"], 1 ,0)


# ENCODING 
#venues encoded
venue = LabelEncoder()
venues  = matches['venue'].unique()

matches['venue_encoded'] = venue.fit_transform(matches['venue'])


#team1 and team2 encoded
team = LabelEncoder()
teams = pd.concat([matches['team1'], matches['team2']]).unique()
team.fit(teams)

matches['team1_encoded'] = team.transform(matches['team1'])
matches['team2_encoded'] = team.transform(matches['team2'])


#toss desision encoded
matches['toss_decision_encoded'] = np.where(matches['toss_decision']=="bat", 1, 0)



# FEATURE ENGINEERING

#win-streak counter- counts the win streak in the last 5 games for every team, resets for the first game of a new season
def wstreak_count(team, date, season, data):
    prev_matches = data[(data['date']< date) & 
                        (data['season']==season) &
                        ((data['team1'] == team) | (data['team2'] == team))]
    last5 = prev_matches.tail(5)
    count = (last5['winner']==team).sum()

    return count

#adding columns for win streaks
matches['team1_last5'] = matches.apply(lambda row: wstreak_count(row['team1'], row['date'], row['season'] ,matches), axis =1)
matches['team2_last5'] = matches.apply(lambda row: wstreak_count(row['team2'], row['date'], row['season'] ,matches), axis =1)

#win-rate counter- calculates the winrate of every team in the current season
def wrate_count(team, date,season,data):
    prev_matches = data[(data['date']< date) & 
                        (data['season']==season) &
                        ((data['team1'] == team) | (data['team2'] == team))]
    
    wins = (prev_matches['winner']==team).sum()
    total = len(prev_matches)
    if total < 3:
        return 0.5
        
    winrate= wins/total
    return winrate 

#adding columns for winrate 
matches['team1_winrate'] = matches.apply(lambda row: wrate_count(row['team1'], row['date'], row['season'] ,matches), axis =1)
matches['team2_winrate'] = matches.apply(lambda row: wrate_count(row['team2'], row['date'], row['season'] ,matches), axis =1)


#head to head: counts the head to head winrate between 2 teams
def h2h(team1, team2, date, data):
    encounters = data[
    (data['date'] < date) &
    (
        ((data['team1']==team1) & (data['team2']==team2)) | 
        ((data['team1']==team2) & (data['team2']==team1))
    )]
    total = len(encounters)
    if total ==0:
        return 0.5
    
    team1_wins = (encounters['winner']==team1).sum()
    return team1_wins/total

#adding the h2h column 
matches['h2h'] = matches.apply(lambda row: h2h(row['team1'], row['team2'], row['date'], matches), axis = 1)

#taking important factors into features, to train the model
features = ['team1_encoded', 'team2_encoded', 'venue_encoded', 'team1_last5', 'team2_last5', 'team1_winrate', 'team2_winrate', 'h2h']
X = matches[features]
y = matches['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# MODEL TRAINING
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_predict))


# PREDICTION 
#predict match: Predicts the result of a game by taking 2 teams and a venue as input
def predict_match(team1_name, team2_name, venue_name):
    team1_encoded = team.transform([team1_name])[0]
    team2_encoded = team.transform([team2_name])[0]
    venue_encoded = venue.transform([venue_name])[0]
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    season = 2026

    team1_last_5 = wstreak_count(team1_name, today, season, matches)
    team2_last_5 = wstreak_count(team2_name, today, season, matches)
    team1_winrate = wrate_count(team1_name, today,season, matches)
    team2_winrate = wrate_count(team2_name, today,season, matches)
    h2h_val = h2h(team1_name, team2_name, today, matches)

    input_features = pd.DataFrame([[team1_encoded, team2_encoded, venue_encoded, team1_last_5, team2_last_5, team1_winrate, team2_winrate, h2h_val]], 
                columns=['team1_encoded', 'team2_encoded', 'venue_encoded', 'team1_last5', 'team2_last5', 'team1_winrate', 'team2_winrate', 'h2h'])
    

    proba = model.predict_proba(input_features)[0]
    winner = team1_name if proba[1] > 0.5 else team2_name
    confidence = max(proba) * 100

    return winner, confidence

correct_predictions = 0
for i, row in matches[matches['season'] == 2026].iterrows():
    predicted, confidence = predict_match(row['team1'], row['team2'], row['venue'])
    if predicted == row['winner']:
        correct_predictions +=1

print(f"2026 accuracy: {correct_predictions/(len(matches[matches['season'] == 2026]))*100:.1f}%")

print(predict_match('Rajasthan Royals', 'Lucknow Super Giants', 'Sawai Mansingh Stadium,Jaipur'))

#EVALUATION 
#plotting 
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind = 'barh')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()



