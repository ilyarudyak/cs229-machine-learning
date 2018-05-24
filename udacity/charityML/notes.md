### table of contents 

- **Exploring the Data.**
- **Preparing the Data.**
- **Evaluating Model Performance.** In this section, we will investigate four 
different algorithms, and determine which is best at modeling the data. 
Three of these algorithms will be supervised learners of your choice, 
and the fourth algorithm is known as a naive predictor.
- **Improving Results.** In this final section, you will choose from 
the three supervised learning models the best model to use on the student data. 
You will then perform a grid search optimization for the model over 
the entire training set (X_train and y_train) by tuning at least one parameter 
to improve upon the untuned model's F-score.
- **Feature Importance.**

### goal

- Your goal with this implementation is to construct a model that accurately 
predicts whether an individual makes more than `$50,000`.


### data exploration

- `n_records=45,222 n_greater_50k=11,208 n_at_most_50k=34,014 greater_percent=24.8%`