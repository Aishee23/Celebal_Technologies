import seaborn as sns
import pandas as pd

# Loading the Titanic dataset
df = sns.load_dataset('titanic')
print(df.head())

# Displaying missing values
print(df.isnull().sum())

# Filling missing 'age' with the mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Filling missing 'embarked' with the mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Dropping 'deck' and 'embark_town' due to too many missing values
df.drop(columns=['deck', 'embark_town'], inplace=True)

# Dropping rows where 'fare' is missing
df.dropna(subset=['fare'], inplace=True)

# Optionally, dropping 'alive' as it is redundant with 'survived'
df.drop(columns=['alive'], inplace=True)

print(df.isnull().sum())


# Convert 'sex' and 'embarked' using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# For 'class' and 'who', we can also use one-hot encoding
df = pd.get_dummies(df, columns=['class', 'who'], drop_first=True)

print(df.head())


from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Scale 'age' and 'fare'
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

print(df.head())


# Create a new feature 'family_size' from 'sibsp' and 'parch'
df['family_size'] = df['sibsp'] + df['parch']

# Create a binary feature indicating whether a passenger is alone
df['is_alone'] = (df['family_size'] == 0).astype(int)

print(df.head())


# No need to encode as it is already in numerical format (0, 1)
print(df['survived'].value_counts())

