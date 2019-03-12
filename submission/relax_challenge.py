import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def get_adopted_users(df):
    adopted = []
    # loop through users
    for user in range(1, 12001):
        is_adopted = False
        user_logins = df[df['user_id'] == user]['value'].values
        # if less than three logins, no adopted
        if len(user_logins) < 3:
            adopted.append(0)
        else:
            # loop through every first and third day and verify if adopted or not
            for i in range(len(user_logins) - 2):
                if (user_logins[i + 2] - user_logins[i]).astype('timedelta64[D]') <= timedelta(days=7):
                    adopted.append(1)
                    is_adopted = True
                    break
            if is_adopted is False:
                adopted.append(0)
    return adopted

DIR = '/Users/rvg/Documents/springboard_ds/relax_challenge/'

# load in dataframes
users = pd.read_csv(DIR + 'takehome_users.csv', encoding='latin-1')
engagement = pd.read_csv(DIR + 'takehome_user_engagement.csv', encoding='latin-1')

# set creation_time
users['creation_time'] = pd.to_datetime(users['creation_time'])
# last_session_creation_time is a unix time stamp, so use map to convert to datetime
users['last_session_creation_time'] = users['last_session_creation_time'].map(lambda data:
datetime.fromtimestamp(int(data)).strftime('%Y-%m-%d %H:%M:%S'),
    na_action='ignore')

# convert to datetime
users['last_session_creation_time'] = pd.to_datetime(users['last_session_creation_time'])
# subtract to find a difference between when account created and last logged in
# sort of a proxy for user activity
users['creation_login_difference'] = users['last_session_creation_time'] - users['creation_time']
# use seconds to make differences more distinct
users['creation_login_difference'] = [x.total_seconds() for x in users['creation_login_difference']]
users['creation_login_difference'] = users['creation_login_difference'].fillna(0)
users.drop(['last_session_creation_time', 'creation_time'], axis=1, inplace=True)

# pick the top six email providers and make them features
users['email_provider'] = [x.split('@')[1] for x in users['email']]
users['count'] = np.ones(len(users))
grouped = users[['email_provider', 'count']].groupby('email_provider').sum().reset_index()
top_six_emails = users.email_provider.value_counts().index[:6]
users['email_provider'] = [email if email in top_six_emails else 'None' for email in users['email_provider']]
users = pd.get_dummies(users, columns=['email_provider'], drop_first=True)
users.drop(['email', 'count'], axis=1, inplace=True)

# make the invited_by_user_id feature a binary variable
users['is_invited'] = [0 if np.isnan(x) == True else 1 for x in users['invited_by_user_id']]
users.drop(['invited_by_user_id'], axis=1, inplace=True)

# split creation source into categorical variable
users = pd.get_dummies(users, columns=['creation_source'], drop_first=True)

# find adopted users and add to dataframe
engagement['time_stamp'] = pd.to_datetime(engagement['time_stamp'], format='%Y-%m-%d %H:%M:%S')
melt_engage = pd.melt(engagement, id_vars='user_id', value_vars='time_stamp')
adopted_users = get_adopted_users(melt_engage)
users['is_adopted'] = adopted_users

# drop unneeded features
users.drop(['object_id', 'name'], axis=1, inplace=True)

print(users.info())
# separate dependent from independent variables
X = users.drop('is_adopted', axis=1)
y = users['is_adopted']

# do test/train split of 20/80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Define function to get metrics of the model

def get_metrics(true_labels, predicted_labels):
    print ('Accuracy: ', accuracy_score(true_labels, predicted_labels))
    print (classification_report(true_labels, predicted_labels))
    return None

rfc = RandomForestClassifier(class_weight='balanced_subsample')
# build model
rfc.fit(X_train, y_train)
# predict using model
y_predict = rfc.predict(X_test)

print('Test set performance:')
get_metrics(true_labels=y_test, predicted_labels=y_predict)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_predict)
pd.DataFrame(cm, index=range(0, 2), columns=range(0, 2))

# Compute predicted probabilities
y_pred_prob = rfc.predict_proba(X_test)[:, 1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, AUC: {:.4f}'.format(roc_auc_score(y_test, y_pred_prob)))
plt.savefig(DIR + 'ROC_curve.png', dpi=350)
plt.close()

# Compute and print AUC score
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_prob)))

# get feature importances
fi = pd.DataFrame(list(zip(X.columns, rfc.feature_importances_)), columns=['features', 'Importance'])
fi.sort_values(by='Importance', ascending=False).head(5)
