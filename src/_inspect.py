import pandas as pd
df = pd.read_csv('archive/matches_aggregated.csv')
inn1 = df[df['innings'] == 1]
print('mean team_runs by season (inn1 only):')
print(inn1.groupby('season')['team_runs'].agg(['mean', 'std', 'count']).round(1))
