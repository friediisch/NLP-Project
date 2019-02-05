from utils import *
import pandas as pd

model3 = pd.read_csv('../data/models/keras/results_model3.csv', sep=';').values # wordvec-docvec
model4 = pd.read_csv('../data/models/keras/results_model4.csv', sep=';').values # tfidf
model5 = pd.read_csv('../data/models/keras/results_model5.csv', sep=';').values # beide

F3 = F_score(model3[:, 3], model3[:, 4])
F4 = F_score(model4[:, 3], model4[:, 4])
F5 = F_score(model5[:, 3], model5[:, 4])

table = []

for mod in [model3, model4, model5]:
    model_measures = []
    for measure in [accuracy, precision, recall, F_score]:
        model_measures.append(measure(mod[:, 3], mod[:, 4]))
    table.append(model_measures)

table = np.array(table)

table = pd.DataFrame(table, columns=['Accuracy', 'Precision', 'Recall', 'F-Score'], index=['W2V', 'tf-idf', 'hybrid'])

table.to_csv('../data/models/evaluation/performance_table.csv', sep=';')

print(table)