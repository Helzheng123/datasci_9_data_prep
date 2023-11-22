import pandas as pd 

## get data 

# original link: https://catalog.data.gov/dataset/nchs-death-rates-and-life-expectancy-at-birth 
# data download link: 
datalink = 'https://data.cdc.gov/api/views/w9j2-ggv5/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.size
df.sample(5)

## save as csv to model_dev1/data/raw/death_life
df.to_csv('model_dev1/data/raw/death_life.csv', index=False)

## save as pickle to model_dev1/data/raw/death_life
df.to_pickle('model_dev1/data/raw/death_life.pkl')
