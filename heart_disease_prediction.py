import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#gathering of data
data=pd.read_csv("C:/Users/91930/Documents/datasetML/Heart_Disease_Prediction.csv")
y=data.target.values
x_data=data.drop(['target'],axis=1)

#preparing of data
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


