from utils import *
import pandas as pd

model3 = pd.read_csv('../data/models/keras/results_model3.csv', sep=';', index_col=False).values # wordvec-docvec
model4 = pd.read_csv('../data/models/keras/results_model4.csv', sep=';', index_col=False).values # tfidf
model5 = pd.read_csv('../data/models/keras/results_model5.csv', sep=';', index_col=False).values # beide
model3_unbalanced = pd.read_csv('../data/models/keras/results_model3_modified_classbalance.csv', sep=';', index_col=False).values # beide
model4_unbalanced = pd.read_csv('../data/models/keras/results_model4_modified_classbalance.csv', sep=';', index_col=False).values # beide
model5_unbalanced = pd.read_csv('../data/models/keras/results_model5_modified_classbalance.csv', sep=';', index_col=False).values # beide


table = []

for mod in [model3, model3_unbalanced, model4, model4_unbalanced, model5, model5_unbalanced]:
    model_measures = []
    for measure in [accuracy, precision, recall, F_score]:
        model_measures.append(measure(mod[:, 2], mod[:, 3]))
    table.append(model_measures)

table = np.array(table)

table = pd.DataFrame(table, columns=['Accuracy', 'Precision', 'Recall', 'F-Score'], index=['W2V_1:1', 'W2V_1:9', 'tf-idf_1:1', 'tf-idf_1:9', 'hybrid_1:1', 'hybrid_1:9'])

table.to_csv('../data/models/evaluation/performance_table.csv', sep=';')

print(table)