import pandas as pd
import os

# Check root directory files
files = ['output10.csv', 'output20.csv', 'output30.csv', 'output40.csv']
print('Root directory files:')
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        ranking_counts = df['ranking'].value_counts().to_dict()
        print(f'{f}: {ranking_counts}')

# Check data directory files
data_dir = 'data/twitter_dataset/group_polarization/'
print('\nData directory files:')
for i in range(10, 90, 10):
    file_path = f'{data_dir}output{i}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        ranking_counts = df['ranking'].value_counts().to_dict()
        print(f'output{i}.csv: {ranking_counts}')
