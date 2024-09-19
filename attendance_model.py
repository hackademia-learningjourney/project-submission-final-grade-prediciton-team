
# Import all necessary libraries required. Prior installation required
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main_module():
    #Reading the Student data file and assigning it to variables
    try:
        df = pd.read_csv('student_performance.csv')
        att_rate = df[['AttendanceRate']]
        final_grade = df['FinalGrade']
    except Exception :
        print("Error while loading data, try again later!")
        return
    #Fitting the model with the data
    try:
        att_rate_train, att_rate_test, final_grade_train, final_grade_test = train_test_split(att_rate, final_grade, test_size=0.2)
        model = LinearRegression()
        model.fit(att_rate_train, final_grade_train)
    except Exception as e:
        print("Error while fitting the model")
        return
    #Taking user input, calculating the grade, checking confidence score of the model
    try:
        test_score = model.score(att_rate_test, final_grade_test)
        attendance = float(input("Enter your attendance days out of 100: "))
        prediction = model.predict(pd.DataFrame([[attendance]]))
        print(f"Your Predicted grade is: {prediction[0]}")
        print(f"The model is: {test_score*100:.3f} % sure about the prediction")
    except Exception as e:
        print("Error while calculation of confidence, try after update!")
#Calling the main function
try:
    main_module()
except Exception:
    print("Error while fetching results. Please try after update!")