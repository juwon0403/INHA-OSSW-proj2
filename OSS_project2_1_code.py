import pandas as pd

df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')


# 1
for year in range(2015, 2019):
    for col in ['H', 'avg', 'HR', 'OBP']:
        top_players = df[df['year'] == year].nlargest(10, col)

        top_players.index = range(1, 11)

        print(f"\nTop 10 players of {year} in {col}: ")
        print(top_players['batter_name'])


# 2
by_position = df[df['year'] == 2018].groupby('cp')['war'].idxmax()
best_players = df.loc[by_position, ['cp', 'batter_name']]

best_players.set_index('cp', inplace=True)
best_players.index.name = None
best_players.columns = ['']

print("\nPlayer with the highest war by position in 2018: ")
print(best_players)


# 3
correlation = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()
highest_corr = correlation['salary'].nlargest(2).index[1]

print("\nThe highest correlation with salary: ")
print(highest_corr)