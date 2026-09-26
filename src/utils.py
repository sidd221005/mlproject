import os 
import sys 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException 
import dill 
from sklearn.metrics import r2_score

try:
    def save_object(file_path,obj):
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
except Exception as e:
    raise CustomException(e,sys)


try:
    def evaluate_models(X_train,y_train,x_test,y_test,models):
        try:
            report={}
            for i in range(len(models)):
                model=list(models.values())[i]
                # Train model
                model.fit(X_train,y_train)

                # Predict Testing data
                y_test_pred=model.predict(x_test)

                # Get r2 score for the model
                test_model_score=r2_score(y_test,y_test_pred)

                report[list(models.keys())[i]]=test_model_score

        
            return report
            
        except Exception as e:
            raise CustomException(e,sys)


except Exception as e:
    raise CustomException(e,sys)