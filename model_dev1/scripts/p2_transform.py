import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev1/data/raw/death_life.pkl')

## get column names
df.columns

## do some data cleaning of column names, 
## make them all lower case, remove white spaces and replace with _ 
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')','')
df.columns

## get data types
df.dtypes # nice combination of numbers and strings/objects 
len(df)

# keep columns 
to_keep = [
    'year',
    'race',
    'sex',
    'average_life_expectancy_years',
    'age-adjusted_death_rate'
]
df = df[to_keep]
print(df)

## perform ordinal encoding on sex
enc = OrdinalEncoder()
enc.fit(df[['sex']])
df['sex'] = enc.transform(df[['sex']])

## create dataframe with mapping for sex
df_mapping_sex = pd.DataFrame(enc.categories_[0], columns=['sex'])
df_mapping_sex['sex_ordinal'] = df_mapping_sex.index
df_mapping_sex

## save mapping to csv
df_mapping_sex.to_csv('model_dev1/data/processed/mapping_sex.csv', index=False)

## perform ordinal encoding on race
enc = OrdinalEncoder()
enc.fit(df[['race']])
df['race'] = enc.transform(df[['race']])

## create dataframe with mapping for race
df_mapping_race = pd.DataFrame(enc.categories_[0], columns=['race'])
df_mapping_race['race_ordinal'] = df_mapping_race.index
df_mapping_race

## save mapping to csv
df_mapping_race.to_csv('model_dev1/data/processed/mapping_race.csv', index=False)

## Save the whole thing
df.to_csv('model_dev1/data/processed/processed_death_life.csv', index=False)