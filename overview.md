### process flow
    - EDA - data definitions, domain knowledge, general trends, dummifying, binarizing
    - simple logistic regression model on 3 features that we thought looked significantly different
    for fraud v. not fraud
    - development of script to compare and cross-validate different classification models
    - based on these results, going back and looking at data for further features

### preprocessing
    - convert org_facebook to binary - greater/less than 5
    - convert org_twitter to binary - greater/less than 5
    - dummify delivery_method
    - N/As are included in the classification process

### assessment metrics selected
    - Model chosen on best profit score
    - Also looked at:
     - recall_score
     - precision_score
     - f1_score
     - accuracy_score

### validation and testing methodology
    - Gridsearch with best model chosen from the cross-validation profit score on training data
    - Calculated other scores on testing set to make sure it makes sense:
     - recall_score, precision_score, f1_score, accuracy_score


### parameter tuning involved in generating the model
    For RandomForestClassifier:
    - tuned tree depth
    - number of trees
    - max depth
    For LogisticRegression:
    - None
    For KNN:
    - number of neighbors
    - distance weighting
    For AdaBoostClassifier:
    - For DecisionTreeClassifier:
     - max depth
     - max features
    - learning rate
    - number of estimators

    Threshold:
    Probability cutoff threshold chosen as the one which gives the best
    profit score.


### further steps you might have taken if you were to continue the project
    - speak with the fraud team to understand the different types of fraud flags
    - go over the data dictionary with the company
    - use NLP on description to try and improve model performance

## flask app
    - Flask app running on an AWS instance
    - call being made to hedoku app to get data every 60s
    - public app which shows all data in mongo db, predicts the probability of Fraud and indicates whether its high or low risk.
    
