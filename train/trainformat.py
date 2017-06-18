import pandas as pd
import glob

allfiles = glob.glob('*.csv')
index = 0

def testing(file):
    file = file.loc[:,'Open':'Volume']
    file = file.values.reshape(1, -1)
    return file


for _fileT in allfiles:
    nFile = pd.read_csv(_fileT, header=0)
    fFile = testing(nFile)
    df = pd.DataFrame(fFile)
    new_df = df.iloc[:500]
    new_df = new_df.shift(1, axis=1)
    new_df.to_csv('HeadCSV/FinalCSV.csv', mode='a', index=False)