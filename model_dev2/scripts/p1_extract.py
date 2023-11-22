import pandas as pd 

## get data 

# original link: https://catalog.data.gov/dataset/nchs-leading-causes-of-death-united-states 
# data download link: 
datalink = 'https://data.cdc.gov/api/views/bi63-dtpu/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.size
df.sample(5)

## save as csv to model_dev2/data/raw/leading_deaths
df.to_csv('model_dev2/data/raw/leading_deaths.csv', index=False)

## save as pickle to model_dev2/data/raw/leading_deaths
df.to_pickle('model_dev2/data/raw/leading_deaths.pkl')