import pandas as pd 

## get data 

# original link: https://catalog.data.gov/dataset/nchs-leading-causes-of-death-united-states 
# data download link: 
datalink = 'https://data.cdc.gov/api/views/bi63-dtpu/rows.csv?accessType=DOWNLOAD'

# This is a dataset that shows the age-adjusted death rates for the 10 leading causes of death in the 50 states and District of Columbia (from 1999-2017).
# Age adjusted death rates (per 100,000 population) are based on the US standard population in 2000.

df = pd.read_csv(datalink)
df.size
df.sample(5)

## save as csv to model_dev2/data/raw/leading_deaths
df.to_csv('model_dev2/data/raw/leading_deaths.csv', index=False)

## save as pickle to model_dev2/data/raw/leading_deaths
df.to_pickle('model_dev2/data/raw/leading_deaths.pkl')