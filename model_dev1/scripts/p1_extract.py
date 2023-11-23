import pandas as pd 

## get data 

# original link: https://catalog.data.gov/dataset/nchs-death-rates-and-life-expectancy-at-birth 
# data download link: 
datalink = 'https://data.cdc.gov/api/views/w9j2-ggv5/rows.csv?accessType=DOWNLOAD'

# This dataset shows the US mortality trends since the 1900s and highlights the differences in age-adjusted death rates and life expectancy at birth by race and sex.
# Age-adjusted death rates (deaths per 100,000) after 1998 are calculated based on the 2000 U.S. standard population.

df = pd.read_csv(datalink)
df.size
df.sample(5)

## save as csv to model_dev1/data/raw/death_life
df.to_csv('model_dev1/data/raw/death_life.csv', index=False)

## save as pickle to model_dev1/data/raw/death_life
df.to_pickle('model_dev1/data/raw/death_life.pkl')
