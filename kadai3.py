# p=0.8

# answer1=10*p**3*(1-p)**2+5*p**4*(1-p)+p**5

# print(answer1)

import scipy.stats as stats

data_list=[2,-5,-4,-8,3,0,3,-6,-2,1,0,-4]

print(stats.shapiro(data_list))