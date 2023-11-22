import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev2/data/raw/leading_deaths.pkl')

## get column names
df.columns

## data cleaning of column names, 
## make them all lower case, remove white spaces and replace with _ 
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## get data types
df.dtypes
len(df)

# keep columns 
to_keep = [
    'year',
    '113_cause_name',
    'cause_name',
    'state',
    'deaths',
    'age-adjusted_death_rate'
]
df = df[to_keep]
print(df)

## perform ordinal encoding on 113_cause_name
enc = OrdinalEncoder()
enc.fit(df[['113_cause_name']])
df['113_cause_name'] = enc.transform(df[['113_cause_name']])

## create dataframe with mapping for 113_cause_name
df_mapping_113_cause_name = pd.DataFrame(enc.categories_[0], columns=['113_cause_name'])
df_mapping_113_cause_name['113_cause_name_ordinal'] = df_mapping_113_cause_name.index
df_mapping_113_cause_name

## save mapping to csv
df_mapping_113_cause_name.to_csv('model_dev2/data/processed/mapping_113_cause_name.csv', index=False)

## perform ordinal encoding on cause_name
enc = OrdinalEncoder()
enc.fit(df[['cause_name']])
df['cause_name'] = enc.transform(df[['cause_name']])

## create dataframe with mapping for race
df_mapping_cause_name = pd.DataFrame(enc.categories_[0], columns=['cause_name'])
df_mapping_cause_name['cause_name_ordinal'] = df_mapping_cause_name.index
df_mapping_cause_name

## save mapping to csv
df_mapping_cause_name.to_csv('model_dev2/data/processed/mapping_cause_name.csv', index=False)

## perform ordinal encoding on cause_name
enc = OrdinalEncoder()
enc.fit(df[['cause_name']])
df['cause_name'] = enc.transform(df[['cause_name']])

## create dataframe with mapping for race
df_mapping_cause_name = pd.DataFrame(enc.categories_[0], columns=['cause_name'])
df_mapping_cause_name['cause_name_ordinal'] = df_mapping_cause_name.index
df_mapping_cause_name

## save mapping to csv
df_mapping_cause_name.to_csv('model_dev2/data/processed/mapping_cause_name.csv', index=False)


## Save the whole thing
df.to_csv('model_dev1/data/processed/processed_death_life.csv', index=False)