import pandas as pd
from sklearn import linear_model as lm
from sklearn.preprocessing import scale, OneHotEncoder

def ncaa_model(col_labels = None, year = 2017, model_type = None):

    data = pd.read_csv('NCAA2001_2017.csv')
    data_2018 = pd.read_csv('NCAA2018.csv')
    data_2018['year'] = 2018
    data = data.append(data_2018)

    col_labels = [              # Stats for higher seed
                'TopFGPer',     # Field goal percentage
                'Top3Per',      # 3 Point Percentage
                'TopAST',       # Assists per game
                'TopEFGPer',    # Effective field goal percentage
                'TopFTR',       # Free throw rate
                'TopTOPer',     # Turnover percentage
                'TopORTG',      # Offensive rating
                'TopDRTG',      # Defensive rating
                'TopSOS',       # Strength of schedule
                'TopORB',       # Offensive rebounds
                'TopDRB',       # Defensive rebounds
                'TopTOPer',     # Turnovers per game
                'TopPTs',       # Points
                'TopOppPTS',    # Opposing Points
                'TopTravel',    # Team Travel Distance
                'BotFGPer',     # Stats for lower seed
                'Bot3Per',
                'BotAST',
                'BotEFGPer',
                'BotFTR',
                'BotTOPer',
                'BotORTG', 
                'BotDRTG',
                'BotSOS',
                'BotORB',
                'BotDRB',
                'BotTOPer',
                'BotPTs',
                'BotOppPTS',
                'BotTravel'
                ]

    # don't scale SeedType
    if 'SeedType' in col_labels:
        col_labels.remove('SeedType')
        if len(col_labels) != 0:
            data[col_labels] = scale(data[col_labels])
        col_labels.insert(0, 'SeedType')
        
    else:
        data[col_labels] = scale(data[col_labels])

    # change SeedTypes to integers in case need to encode later
    data = data.replace(
            ['OneSixteen', 'TwoFifteen', 'ThreeFourteen',
                'FourThirteen', 'FiveTwelve', 'SixEleven',
                'SevenTen', 'EightNine'],
            [1, 2, 3, 4, 5, 6, 7, 8])

    train = data.loc[(data['year'] != year) & 
            (data['year'] != 2018)][col_labels]
    train_results = data.loc[(data['year'] != year) & 
            (data['year'] != 2018)]['Upset'] # not a df

    test = data.loc[data['year'] == year][col_labels]
    results_columns = ['SeedType', 'TopSeed', 'BotSeed', 'Upset']
    test_results = data.loc[data['year'] == year][results_columns]

    # One-hot encode the seeding type
    enc = OneHotEncoder(categorical_features = [0])
    train = enc.fit_transform(train).toarray()
    test = enc.fit_transform(test).toarray()

    model = lm.LogisticRegression()
    model.fit(train, train_results.as_matrix())

    predictions = model.predict_proba(test)
    proba = []
    for i in range(len(predictions)):
        proba.append(predictions[i][1]) # second column is upset percentage

    test_results['UpsetProba'] = proba

    return test_results



