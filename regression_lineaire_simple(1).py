#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Python implementation of the simple_linear_regression technique on our small dataset

#import libraries

import numpy as np
import matplotlib.pyplot as plt


# In[43]:


#function to determine the best coefficients of the simple linear regression hypothesis function

def estimate_coef(x,y):
    #number of observation/points
    n=np.size(x)
    #mean of x and y vector
    m_x=np.mean(x)
    m_y=np.mean(y)
    #calculate the covariance(x,y) and variance(x)
    #cov(x,y)=mean(x.y)-mean(x).mean(y)
    cov_xy=np.mean(x*y)-np.mean(x)*np.mean(y)
    #var_x=np.mean(x**2)-m_x**2
    #var_y=np.mean(y**2)-m_y**2
    #caculating regression coefficients
    a= cov_xy/np.var(x)
    b=m_y -a*m_x
    r=cov_xy/(np.std(x)*np.std(y))
    return(a,b,r)
#data visualization function
def plot_regression_line(x,y,b):
    #plotting the actual points as scatter plot
    plt.scatter(x,y,color="m",marker="o")
    #predicted reponse vector
    y_pred=b[0]+b[1]*x
    #plotting the regression line 
    plt.plot(x,y_pred,color="g")
    #putting labels
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(labels=("cloud of point","model_ linear_regressiom"))
    # function to show plot
    plt.show()
#calculate the linear correlation coefficient

    
def main():
    # observation / data
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,3,2,5,7,8,8,9,10,12])
    # estimating coefficients
    b=estimate_coef(x,y)
    print("Estimated coefficients: \n a=",b[1],"\n b=",b[0]," \n linear correlation coefficient r=",b[2])
    
   
          
          
          
    #plotting regression line
    plot_regression_line(x,y,b)

if __name__ == "__main__":
    main()    
  
    
        
          
    
    


# In[ ]:





# In[ ]:




