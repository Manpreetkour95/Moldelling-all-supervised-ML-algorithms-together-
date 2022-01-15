# Moldelling-all-supervised-ML-algorithms-together-
The dataset presented in this case is of Semiconductor manufacturing process. It represents a selection of such features where each example represents a single production entity withassociated measured features and the labels represent a simple pass/fail yield for in house line testing.

Using for all ML models has been applied to the data.
A glimpse of code is given below-

models = []

models.append(('LR', LogisticRegression()))
models.append(('XGB', XGBClassifier(random_state=1, eval_metric='mlogloss'))) 
models.append(('KNN', KNeighborsClassifier(5)))
models.append(('DT', DecisionTreeClassifier(max_depth=5)))
models.append(('RF', RandomForestClassifier(n_estimators=196)))
models.append(('ADA', AdaBoostClassifier()))
models.append(('SVC', SVC()))


#evaluate each model in turn

results = []
names = []
scoring = 'accuracy'
precision=[]
recall=[]


for name, model in models:
    
    clf = model.fit(X_train_res, y_train_res)
    pred = clf.predict(X_test)
    
    cv_results=accuracy_score(y_test, pred)


    tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
        #visualization of confusion matrix in the form of a heatmap
        
    cm= confusion_matrix(y_test, pred)
