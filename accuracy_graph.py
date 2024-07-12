import matplotlib.pyplot as plt
import numpy as np

DT = 0.9999890964352158
KNN = 0.9999291268289027
NB = 0.9970165120859201
GB= 0.9997710251395315
RF = 0.9999863705440197
ET = 0.9999918223264118
XGB = 0.999995911163206
ADA = 0.999841898310629



# Accuracy scores for each model
accuracy_scores = [DT, KNN, NB, GB, RF, ET, XGB, ADA]


# Model names
models = ['Decision Tree', 'KNN', 'Naive Bayes', 'Gradient Boosting', 'Random Forest', "Extra Trees", "XGB", 'Adaboost']

sorted_data = sorted(zip(models, accuracy_scores), key=lambda x: x[1], reverse=True)
sorted_models, sorted_accuracy_scores = zip(*sorted_data)



# Create figure and axes
plt.barh(sorted_models, sorted_accuracy_scores)
plt.xlim(0.99, 1)
plt.title('Accuracy Comparison of Different ML Models')
plt.xlabel('Accuracy')
plt.show()






