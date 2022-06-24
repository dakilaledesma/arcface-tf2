import os
from sklearn.metrics import accuracy_score
result_file = open("results.csv")
result_lines = result_file.readlines()
rename_file = open("rename_results.csv")
rename_lines = rename_file.readlines()

cat_dict = {int(v): int(k) for k, v in [z.strip().split(',') for z in rename_lines]}

y_pred = []
y_true = []
for line in result_lines:
    line = line.strip().split(',')
    pred = int(line[0])
    cat = int(line[1].split('/')[-2])
    y_pred.append(cat_dict[pred])
    y_true.append(cat)

print(accuracy_score(y_pred, y_true))
