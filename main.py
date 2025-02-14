import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import plotly.figure_factory as ff
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score, matthews_corrcoef

file_name = "Data_processed.xlsx"
current_dir = os.getcwd()
file_path = os.path.join(current_dir, file_name)
df = pd.read_excel(file_path)

df['Longitude'] = df['Longitude'].astype(str).replace(r'\s+', '', regex=True).astype(float)

#** IMPRUTER WITH KNN **
imputer = KNNImputer(n_neighbors=3)
df['DaysFromHerbicideToHarvest'] = imputer.fit_transform(df[['DaysFromHerbicideToHarvest']])
df['DaysFromSowingToHerbicide'] = imputer.fit_transform(df[['DaysFromSowingToHerbicide']])
df['HerbicideWeekNum'] = imputer.fit_transform(df[['HerbicideWeekNum']])
df['HerbicideDay'] = imputer.fit_transform(df[['HerbicideDay']])
df['HerbicideMonth'] = imputer.fit_transform(df[['HerbicideMonth']])
df['HerbicideYear'] = imputer.fit_transform(df[['HerbicideYear']])
df['Longitude'] = imputer.fit_transform(df[['Longitude']])

#---------------------------------- DATA PREPROCESSING - BALANCING ---------------------------------
warnings.filterwarnings('ignore')
df['GrainYield'] = df['GrainYield'].apply(lambda x: 1 if x == "A" else 0)

X = df.drop(columns=['GrainYield'])
y = df['GrainYield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

#NORMALIZATION
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)


X_train = pd.DataFrame(X_train_normalized, columns=X.columns)
X_train['GrainYield'] = y_train
df = pd.DataFrame(X_train, columns=list(X.columns) + ['GrainYield'])


#* FEATURE SELECTION *
print("----------------------- FEATURE SELECTION WITH RFECV ---------------------")
X = df.drop('GrainYield', axis=1)
y = df['GrainYield']

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(5)
selector = RFECV(estimator=rf_classifier, step=1, cv=cv, scoring='accuracy')
selector=selector.fit(X, y)

print("Optimal Feature Number:", selector.n_features_)
print("Selected Features:", X.columns[selector.support_])

df_selectedfeatures = pd.DataFrame()
for feature in X.columns[selector.support_]:
    df_selectedfeatures[feature] = df[feature]

X=df_selectedfeatures
y=df['GrainYield']

print("----------------------- TEST AND EVALUATION -------------------------")

lr_classifier = ExtraTreesClassifier(random_state=42)

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []

for train_index, test_index in stratified_kfold.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    lr_classifier.fit(X_train, y_train)

    y_pred = lr_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)

    print("Fold Accuracy:", accuracy)

print("\nAverage Accuracy:", sum(fold_accuracies) / len(fold_accuracies))


print("----------------------- ROC CURVE, AUC, CA, F1, PRECISION, RECALL AND MCC -------------------------  ")


def evaluate_classifiers(X, y):


    warnings.filterwarnings('ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifiers = {
        'Bayesian': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbosity=-1),
        'Extra Trees': ExtraTreesClassifier(random_state=42)
    }


    results = {}
    plt.figure(figsize=(16, 12))

    for idx, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_test)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.subplot(4, 3, idx + 1)
        plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {auc_score:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {clf_name}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')

        # ACCURACY AND OTHER METRICS
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        results[clf_name] = {
            'AUC': auc_score,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'MCC': mcc
        }

        if clf_name == 'Extra Trees':
            extratrees_y_pred = y_pred

    if extratrees_y_pred is not None:
        cm = confusion_matrix(y_test, extratrees_y_pred)
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Viridis'
        )

        fig.update_layout(
            title='Confusion Matrix - Extra Trees',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            yaxis=dict(scaleanchor="x"),
        )

        fig.show()

    plt.tight_layout()
    plt.show()

    return results

def print_results(results):
    for clf_name, metrics in results.items():
        print(f"----- {clf_name} -----")
        for metric_name, value in metrics.items():
            if isinstance(value, list):
                print(f"{metric_name}: \n{value}")
            else:
                print(f"{metric_name}: {value}")
        print()


results = evaluate_classifiers(X, y)
print_results(results)


