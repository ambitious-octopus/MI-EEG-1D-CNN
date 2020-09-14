# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:18:12 2020

@author: xphid
"""


import pandas as pd

import statsmodels.api as sm

from statsmodels.formula.api import ols    # Ordinary Least Squares (OLS) model


# C3 right and left

dfC3=pd.read_csv("C:\\Users\\xphid\\Desktop\\reco\\C3_deltas_mean_sub.csv")


dfC3.boxplot(column=['right', 'left'], grid=False)


modelC3 = ols('right ~ left', data=dfC3).fit()

tableanovaC3=sm.stats.anova_lm(modelC3)

print(tableanovaC3)


#model_C_C3=ols("right ~ C(left)", data= dfC3).fit()
#tableanova_C_C3=sm.stats.anova_lm(modelC3)
#print(tableanova_C_C3)

#%%

# C4 left and right

dfC4=pd.read_csv("C:\\Users\\xphid\\Desktop\\reco\\C4_deltas_mean_sub.csv")


dfC4.boxplot(column=['right', 'left'], grid=False)


modelC4= ols('left ~ right', data=dfC4).fit()

tableanovaC4=sm.stats.anova_lm(modelC4)

print(tableanovaC4)


#model_C_C4=ols("left ~ C(right)", data= dfC4).fit()
#tableanova_C_C4=sm.stats.anova_lm(modelC4)
#print(tableanova_C_C4)