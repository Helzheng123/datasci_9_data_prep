# datasci_9_data_prep
Focus on selecting datasets suitable for a machine learning experiment, with an emphasis on data cleaning, encoding, and transformation steps necessary to prepare the data.

## 1. Dataset Selection:
Description of each Dataset:
 - **[NCHS - Death rates and life expectancy at birth](https://github.com/Helzheng123/datasci_9_data_prep/blob/main/model_dev1/data/raw/death_life.csv)**: This dataset shows the US mortality trends since the 1900s and highlights the differences in age-adjusted death rates and life expectancy at birth by race and sex. Age-adjusted death rates (deaths per 100,000) after 1998 are calculated based on the 2000 U.S. standard population.
 - **[NCHS - Leading Causes of Death: United States](https://github.com/Helzheng123/datasci_9_data_prep/blob/main/model_dev2/data/raw/leading_deaths.csv)**: This is a dataset that shows the age-adjusted death rates for the 10 leading causes of death in the 50 states and the District of Columbia (from 1999-2017). Age-adjusted death rates (per 100,000 population) are based on the US standard population in 2000.

## 2. Data Cleaning and Transformation Plan:
 - In the *Death rates and life expectancy at birth dataset*, I chose to do a classification for the column **sex**. The predictors are stored in X and are created by dropping the **sex** column while the target variable is stored in y which contains the **sex** column from the dataframe.
 - Steps for cleaning and transforming the data are shown here:
```
# this is to replace the white space with _ and ( & ) with white space.
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')','')
df.columns

df.dtypes
len(df)

to_keep = [
    'year',
    'race',
    'sex',
    'average_life_expectancy_years',
    'age-adjusted_death_rate'
]
df = df[to_keep]
print(df)
```
 - I removed the missing values with this:
```
df.dropna(inplace=True)
len(df)
```
 - I encoded the categorical variables with this:
```
enc = OrdinalEncoder()
enc.fit(df[['sex']])
df['sex'] = enc.transform(df[['sex']])

df_mapping_sex = pd.DataFrame(enc.categories_[0], columns=['sex'])
df_mapping_sex['sex_ordinal'] = df_mapping_sex.index
df_mapping_sex

df_mapping_sex.to_csv('model_dev1/data/processed/mapping_sex.csv', index=False)
```
This makes Both sexes as '0', Female as '1', and Male '2'. You can view the data dictionary in [this file](https://github.com/Helzheng123/datasci_9_data_prep/blob/main/model_dev1/data/processed/mapping_sex.csv).

 - In the *Leading Causes of Death - US dataset*, I chose to do a regression for the column **deaths**. The predictors are stored in X and are created by dropping the **deaths** column while the target variable is stored in y which contains the **deaths** column from the dataframe.
 - Steps for cleaning and transforming the data are shown here:
```
# this is the replace the white space with _
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

df.dtypes
len(df)

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
```
 - I removed the missing values with this:
```
df.dropna(inplace=True)
len(df)
```
 - I encoded the categorical variables with this:
```
enc = OrdinalEncoder()
enc.fit(df[['113_cause_name']])
df['113_cause_name'] = enc.transform(df[['113_cause_name']])

df_mapping_113_cause_name = pd.DataFrame(enc.categories_[0], columns=['113_cause_name'])
df_mapping_113_cause_name['113_cause_name_ordinal'] = df_mapping_113_cause_name.index
df_mapping_113_cause_name

df_mapping_113_cause_name.to_csv('model_dev2/data/processed/mapping_113_cause_name.csv', index=False)
```
You can view the processed data for the column [113 cause name here](https://github.com/Helzheng123/datasci_9_data_prep/blob/main/model_dev2/data/processed/mapping_113_cause_name.csv).

**NOTE** I didn't drop any columns as both datasets are small and didn't have many missing values.
