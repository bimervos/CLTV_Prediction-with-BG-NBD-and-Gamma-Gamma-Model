


##################    Load Dataset    ##################

#Import the libraries :

import seaborn as sns
import pandas  as pd
import datetime as dt
from sklearn.preprocessing  import minmax_scale
import matplotlib.pyplot as plt
#!pip install lifetimes
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

#Some configurations

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' %x)

#Load the dataset :

df= pd.read_csv('customer_relationship.csv')

##################    About Dataset    ##################

# The dataset contains sales data of a Turkey-based company that sells tobacco products.
# The data covers a period of 2 years from 2020 to 2022.
# The company has several product categories, including cigarettes, cigars, and pipe tobacco, among others.
# The dataset contains information on sales volume, revenue, and profit for each product category and year.
# The dataset could be used to identify trends and patterns in the company's sales, as well as to perform forecasting and optimization analyses.
#
# In this project, we will perform a descriptive analysis to show how well or poorly sales are going in the company,
# We will measure customer engagement through a cohort analysis, try to calculate the earnings for the coming months and analyze the customers who will bring the most profit to the company.
# Additionally, we will learn the importance of cleaning and preprocessing data prior to conducting any analysis.
#
# Dataset has 6 columns:
#
# invoiceID: Unique Invoice number
# invoice_date: Date of purchase
# customerID: Unique Customer number
# country: Country of purchase
# quantity: Quantity of products purchased
# amount: Total amount of products purchased


##################    1)Data Preparation    ##################

def check_df(dataframe, head=5):
    print(" SHAPE ".center(70, '-'))
    print('Rows: {}'.format(dataframe.shape[0]))
    print('Columns: {}'.format(dataframe.shape[1]))
    print(" TYPES ".center(70, '-'))
    print(dataframe.dtypes)
    print(" MISSING VALUES ".center(70, '-'))
    print(dataframe.isnull().sum())
    print(" DUPLICATED VALUES ".center(70, '-'))
    print(dataframe.duplicated().sum())
    print(" DESCRIBE ".center(70, '-'))
    print(dataframe.describe().T)

check_df(df)

# We see that some customer numbers are missing.
# Analyzing without a customer number would not make sense.
# Therefore, we are removing the missing values from the dataset.

df.dropna(inplace=True)

# We will convert the variable containing date to the date type and we will set the analysis date as one day after the last date in the dataset.

df.loc[:, 'invoice_date']= df.loc[:, 'invoice_date'].apply(pd.to_datetime)

df['invoice_date'].max()
today_date= dt.datetime(2021,12,10)

#When we examine the type of the 'amount' variable, we see that it is an object.

df['amount'] = df['amount'].str.replace(',', '.')
df['amount'] = df['amount'].astype(float)
df.describe().T

# As we can see in the Describe function, there are sudden increases in the maximum points of 'amount' and 'quantity' variables.
# We can also visualize these outliers with a graph:

plt.subplot(1,2,1, xlabel='amount')
sns.boxplot(data=df['amount'], flierprops={"marker": "x"})
plt.subplot(1,2,2, xlabel='quantity')
sns.boxplot(data=df['quantity'], flierprops={"marker": "x"})
plt.show(block=True)

#Let's define and apply a function that removes the outliers from the dataset:

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, 'quantity')
replace_with_thresholds(df, 'amount')
df.describe().T


# Return invoice amounts are named as (-). Filter out the return invoices.
# Let's also filter out the ones with 'amount' value of 0 :

df= df[(~df['invoiceID'].str.contains('C')) & (df['amount'] > 0 )]


##################    2)Preparation of CLTV metrics     ##################

#
# CLTV, also known as Customer Lifetime Value, is a metric system used to estimate the total revenue that a company generates from a customer throughout their lifetime.
#
# Recency refers to the time between a customer's first and last purchase, and is expressed on a weekly basis.
# T value (customers age) indicates the time between the customer's first purchase and the analysis date. It is expressed in weeks.
# Frequency refers to how many times a customer makes a purchase, and can be used to evaluate customer loyalty. (f > 1)
# Monetary represents how much money a customer spends on a company's products or services.It will be expressed as average spending per purchase.
#
# By using these three metrics together, it is possible to estimate how valuable a customer is and help the company create customer-focused strategies.

cltv = df.groupby('customerID').agg({ 'invoice_date': [lambda x: ((x.max()- x.min()).days) / 7 ,
                                                       lambda x: ((today_date- x.min()).days) / 7 ],
                                      'invoiceID': lambda x: x.nunique(),
                                      'amount': lambda x: x.sum() })
cltv.columns= cltv.columns.droplevel(0)
cltv.columns = ['recency', 'T', 'frequency', 'monetary']

#The frequency value should be greater than 1. (There should be at least one purchase):
#Monetary is expressed as the average amount spent per purchase:

cltv = cltv[cltv['frequency']>1]
cltv['monetary'] = cltv['monetary'] / cltv['frequency']
cltv.head()

##################    3)Preparation of BG-NBD Model     ##################

#Let's build the BGF model.

bgf= BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv['frequency'],
        cltv['recency'],
        cltv['T'])

#Expected purchase: (1 week)
#We can use also 'predict' instead of 'conditional_expected_number_of_purchases_up_to_time'

cltv['expected_purchase'] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                                                    cltv['frequency'],
                                                                                    cltv['recency'],
                                                                                    cltv['T'])
cltv.head()

#Top 10 customers with the highest expected purchases within 1 week:

cltv.sort_values(by='expected_purchase',ascending=False).head(10)



##################    4)Preparation of Gamma-Gamma Model     ##################


ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv['frequency'], cltv['monetary'])

#Expected profit (1 week)

cltv['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                          cltv['monetary'])

#Top 10 customers with the highest expected profit within 1 week:

cltv.sort_values(by='expected_average_profit',ascending=False).head(10)

##################    5)CLTV Calculation    ##################

#Let's calculate the CLTV value. With the number entered as the time argument, the desired monthly expected revenue can be calculated.

cltv['cltv'] = ggf.customer_lifetime_value(bgf,
                                   cltv['recency'],
                                   cltv['T'],
                                   cltv['frequency'],
                                   cltv['monetary'],
                                   time=3,
                                   freq='W',
                                   discount_rate=0.01)

#Top 10 customers with the highest expected revenue within 1 month:

cltv.sort_values(by='cltv', ascending=False).head(10)

##################    6)Preparation of Segments    ##################

#By segmenting customers, strategies can be developed by segments:

cltv['segment']  = pd.qcut(cltv['cltv'], 4, labels=['D', 'C', 'B', 'A'])
cltv.groupby('segment').agg({'mean', 'count', 'sum'})


