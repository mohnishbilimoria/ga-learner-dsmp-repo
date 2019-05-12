# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data['Rating'].hist(bins=3)

data = data[data['Rating'] <= 5]
data['Rating'].hist(bins=3)

#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())
missing_data = pd.concat([total_null, percent_null], keys=['Total', 'Percent'],axis=1)
print(missing_data)

data.dropna(inplace=True)

total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())
missing_data_1 = pd.concat([total_null_1, percent_null_1], keys=['Total', 'Percent'],axis=1)
print(missing_data_1)


# code ends here


# --------------

#Code starts here
g = sns.catplot(x="Category", y="Rating", data=data , kind="box", height = 10);
g.set_xticklabels(rotation=90)
g.fig.suptitle('Rating vs Category [BoxPlot]')

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

#Code starts here
print(data['Installs'])
data['Installs'] = data['Installs'].map(lambda x: x.lstrip(',+').rstrip(',+')).str.replace(',','').astype(np.int64)

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

g = sns.regplot(x="Installs", y="Rating", data=data)
g.set_title('Rating vs Installs [RegPlot]')

#Code ends here



# --------------
#Code starts here
print(data['Price'])
data['Price'] = data['Price'].str.replace('$','').astype(np.float64)

g = sns.regplot(x="Price", y="Rating", data=data)
g.set_title('Rating vs Price [RegPlot]')

#Code ends here


# --------------

#Code starts here
print(data['Genres'].unique())
data["Genres"]= data["Genres"].str.split(";", expand = True)[0]

gr_mean = data.groupby(['Genres'], as_index=False).agg({'Rating':'mean'})

print(gr_mean.describe())

gr_mean = gr_mean.sort_values('Rating')

print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])

#Code ends here


# --------------

#Code starts here
print(data['Last Updated'])

data['Last Updated'] = pd.to_datetime(data['Last Updated'])

max_date = data['Last Updated'].max()

data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

g = sns.regplot(x="Last Updated Days", y="Rating", data=data)
g.set_title('Rating vs Last Updated [RegPlot]')

#Code ends here


