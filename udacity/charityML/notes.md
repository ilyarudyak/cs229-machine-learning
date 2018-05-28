### table of contents 

- **Exploring the Data.** 
    - The total number of records, `'n_records'`
    - The number of individuals making more than $50,000 annually, `'n_greater_50k'`.
    - The number of individuals making at most $50,000 annually, `'n_at_most_50k'`.
    - The percentage of individuals making more than $50,000 annually, `'greater_percent'`.

- **Preparing the Data.**
    - Transforming skewed continuous features (`'capital-gain', 'capital-loss'`).
    - Normalizing numerical features (using `sklearn.preprocessing.MinMaxScaler`).
    - One-hot encoding of categorical features (using `pandas.get_dummies()`).

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

### goal and metric

- Your goal with this implementation is to construct a model that accurately 
predicts whether an individual makes more than `$50,000`.


### data exploration

- n_records=`45,222` n_greater_50k=`11,208` n_at_most_50k=`34,014` 
greater_percent=`24.8%`

### algorithms comparison

- accuracy score (for 10% sample): 
    - naive classifier = `24.8%` (== greater_percent)
    - decision tree (min_samples_split=`300`) accuracy = `85.1%` beta .5 = `70.9%` 
    - gaussian NB accuracy = `36.6%` beta .5 = `32.0%`
    - svm accuracy = `83.3%` beta .5 = `67.1%`

### plans

#### decision trees
- plot validation curves for DT: for max_depth and min_samples_split;
first with accuracy and then with f1; --DONE
- optimize DT model on 10% subset;
- plot learning curves for DT;
- what should we do with features, data volume?
- question: what about balance between recall and precision; can we use
F1 score, not F.5?

#### more models
- do logistic regression and random forest;
- plot validation curves for SVM;
- try to optimize SVM on 10% data set; can we beat DT?
- listen Ng about SVM and try to implement it;

#### misc
- do timing of all algorithms on a small subset; 

#### questions
- skewed features;
- one hot encoding alternatives;
- question: what about balance between recall and precision; can we use
F1 score, not F.5?
