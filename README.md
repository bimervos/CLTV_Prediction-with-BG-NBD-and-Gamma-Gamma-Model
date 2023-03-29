# CLTV_Prediction-with-BG-NBD-and-Gamma-Gamma-Model

The dataset contains sales data of a Turkey-based company that sells tobacco products.
The data covers a period of 2 years from 2020 to 2022.
The company has several product categories, including cigarettes, cigars, and pipe tobacco, among others.
The dataset contains information on sales volume, revenue, and profit for each product category and year.
The dataset could be used to identify trends and patterns in the company's sales, as well as to perform forecasting and optimization analyses.

In this project, we will perform a descriptive analysis to show how well or poorly sales are going in the company,
We will measure customer engagement through a cohort analysis, try to calculate the earnings for the coming months and analyze the customers who will bring the most profit to the company.
Additionally, we will learn the importance of cleaning and preprocessing data prior to conducting any analysis.


Dataset has 6 columns:


invoiceID: Unique Invoice number  
invoice_date: Date of purchase  
customerID: Unique Customer number  
country: Country of purchase  
quantity: Quantity of products purchased  
amount: Total amount of products purchased

BG-NBD and Gamma-Gamma Model
BG-NBD (Beta Geometric/Negative Binomial Distribution) and Gamma-Gamma models are two statistical models used for calculating customer lifetime value (CLTV). These models take into account several variables to provide more accurate results for estimating customer lifetime value.

The BG-NBD model analyzes customer purchase behavior and predicts how often customers are likely to make future purchases. It also predicts when customers are likely to churn. The Gamma-Gamma model, on the other hand, predicts how much customers will spend on each purchase transaction.

Using these two models, the CLTV formula can be expanded as follows:

CLTV = (Purchase Frequency using BG-NBD * Average Order Value using Gamma-Gamma) - Customer Costs

In this formula, the BG-NBD model is used to estimate customer purchase frequency, while the Gamma-Gamma model is used to estimate the average order value. Customer costs include all costs associated with acquiring and serving customers, such as customer acquisition costs, marketing costs, and customer service costs.

The BG-NBD and Gamma-Gamma models are effective methods for calculating CLTV. They can be used to understand customer behavior and offer personalized incentives to customers.
