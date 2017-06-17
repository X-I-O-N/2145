import pandas as pd
import glob

allfiles = glob.glob('*.csv')
index = 0

def testing(file):
    file.drop('Gmt time', axis=1)
    file = file.values.reshape(1, -1)
    return file


for _fileT in allfiles:
    index += 1
    nFile = pd.read_csv(_fileT)
    fFile = testing(nFile)
    df = pd.DataFrame(fFile)
    new_df = df.iloc[500, :]
    df.to_csv('HeadCSV/FinalCSV.csv', mode='a', index=True)