# Problem 7: Write a program to construct aBayesian network considering medical data. 
#Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set. 
#You can use Java/Python ML library classes/API.


import pandas as pd
data=pd.read_csv("heart_disease.csv")
heart_disease=pd.DataFrame(data)
print(heart_disease)

from pgmpy.models import BayesianModel
model=BayesianModel([
('age','Lifestyle'),
('Gender','Lifestyle'),
('Family','heartdisease'),
('diet','cholestrol'),
('Lifestyle','diet'),
('cholestrol','heartdisease'),
('diet','cholestrol')
])

from pgmpy.estimators import MaximumLikelihoodEstimator
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)


from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)

print('For age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4')
print('For Gender Enter Male:0, Female:1')
print('For Family History Enter yes:1, No:0')
print('For diet Enter High:0, Medium:1')
print('for lifeStyle Enter Athlete:0, Active:1, Moderate:2, Sedentary:3')
print('for cholesterol Enter High:0, BorderLine:1, Normal:2')

q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age':int(input('enter age')),
    'Gender':int(input('enter Gender')),
    'Family':int(input('enter Family history')),
    'diet':int(input('enter diet')),
    'Lifestyle':int(input('enter Lifestyle')),
    'cholestrol':int(input('enter cholestrol'))
    })

print(q['heartdisease'])
