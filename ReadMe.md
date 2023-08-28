# <span style="color: red;">A CNN-MLP model for Human Activity Recognition.</span>

WISDM dataset is included in the repository, and link to UCI HAR is included also in the descriptiopn part.

To execute the code, you must:

1. Charge the datasets, and then put the correct path to the used one(datasets are included in the Data directory).
2. Choose the adequate "Preprocessing" part.

## For UCI HAR Dataset:

```python
print('No duplicates in our training data: {}'.format(sum(train.duplicated())))
print('No of duplicates in our test data: {}'.format(sum(test.duplicated())))
print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))
print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape, test.shape)

columns = train.columns
columns = columns.str.replace('[()]','')
columns = columns.str.replace('[-]', '')
columns = columns.str.replace('[,]','')
train.columns = columns
test.columns = columns
....
```

## For WISDM Dataset:
```python

label_encode = LabelEncoder()
df['activityLabel'] = label_encode.fit_transform(df['activity'].values.ravel())
interpolation_fn = interp1d(df['activityLabel'] ,df['Z'], kind='linear')
null_list = df[df['Z'].isnull()].index.tolist()
for i in null_list:
    y = df['activityLabel'][i]
    value = interpolation_fn(y)
    df['Z']=df['Z'].fillna(value)
    print(value)
df_train['X'] = (df_train['X']-df_train['X'].min())/(df_train['X'].max()-df_train['X'].min())
df_train['Y'] = (df_train['Y']-df_train['Y'].min())/(df_train['Y'].max()-df_train['Y'].min())
df_train['Z'] = (df_train['Z']-df_train['Z'].min())/(df_train['Z'].max()-df_train['Z'].min())
df_train
......
```

3. Execute the code regularly
