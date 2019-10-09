# Linear Regression 

The dataset for this machine learning project is created from data provided by UNICEF’s State of the World’s Children 2013 report:
http://www.unicef.org/sowc2013/statistics.html <br />  
Child mortality rates (number of children who die before age 5, per 1000 live births) for 195 countries, and a set of other indicators are included.

• Target value: (Under-5 mortality rate (U5MR) 2011)
• Input features: Major factors in these areas (eg: Annual no. of under-5 deaths (thousands) 2011).
• Training data: 100 countries(Afghanistan to Luxembourg).
• Testing data: 95 countries(Madagascar to Zimbabwe).
• Cross-validation: 10 folds Cross-validation.

Error function: $E(w) =\frac{1}{2}\left( \sum_{k=1}^n t_n - w^Tx_n \right)^2$

***Polynomial basis function regression*** 


unregularized for degree 1 to degree 6 polynomials. Include bias term.

![alternativetext](Graph/5.2.1_BeforeNormalize.png)


***Sigmoid basis function regression*** 
<br />  

a single input feature(GNI per capita (US$) 2011', 'Life expectancy at birth (years) 2011), with µ = 100 and s = 2000.0. Include a bias term.

![alternativetext](Graph/5.3_lambda=100.png)

***L2-regularized regression***

 Fit a degree 2 polynomial using λ = {0, .01, .1, 1, 10, $10^2$ , $10^3$, $10^4$}. Use normalized features as input. Include a bias term. Use 10-fold cross-validation to decide on the best value for λ.

 ![alternativetext](Graph/5.4.png)

 the best $\lambda $ should aound $10^2$ to $10^3$.