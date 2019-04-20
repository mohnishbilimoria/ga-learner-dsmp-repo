# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here
data = pd.read_csv(path)
data_sample = data.sample(n=sample_size, random_state=0)

sample_mean = round(np.mean(data_sample.installment),2)
sample_std = round(data_sample['installment'].std(),2)

margin_of_error = round(z_critical * (sample_std / math.sqrt(sample_size)),2)
print('margin of error' , margin_of_error)


confidence_interval = [sample_mean - margin_of_error, sample_mean + margin_of_error]
print('confidence interval', confidence_interval)

true_mean = round(np.mean(data.installment),2)
print('True Mean :',true_mean)


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig, axes = plt.subplots(nrows=3, ncols=1)

for i in range(len(sample_size)):
    m = []
    for j in range(1000):
        m.append(round(np.mean(data.sample(n=sample_size[i], random_state=0).installment),2))
    mean_series = pd.Series(m)
    axes[i].hist(mean_series, bins=10)


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = (data['int.rate'].str.replace("%", "").str.strip()).astype('float64')

data['int.rate'] = data['int.rate'] / 100

z_statistic, p_value = ztest(data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')

# print z statistic and p value
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)

# check the p-value
if p_value < 0.05:
    inference = 'Reject'
else :
    inference = 'Accept'

print(inference)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest


#Code starts here
z_statistic, p_value = ztest(data[data['paid.back.loan']=='No']['installment'],data[data['paid.back.loan']=='Yes']['installment'])

# print z statistic and p value
print("Z-statistics = ",z_statistic)
print("p-value = ",p_value)

# check the p-value
if p_value < 0.05:
    inference = 'Reject'
else :
    inference = 'Accept'

print(inference)


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()

no = data[data['paid.back.loan']=='No']['purpose'].value_counts()

observed = pd.concat([yes.transpose(),no.transpose()], axis=1, keys= ['Yes','No'])

chi2, p, dof, ex = chi2_contingency(observed)

print('chi2', chi2)
print('critical_value', critical_value)

if(chi2 > critical_value):
    print('Reject')
else:
    print('Accept')


