import os
result_file = open("arcface_results.csv")
result_lines = result_file.readlines()
rename_file = open("rename_results.csv")
rename_lines = rename_file.readlines()

cat_dict = {v: k for k, v in [z.strip().split(',') for z in rename_lines]}

for line in result_lines:
    line = line.strip().split(',')
    pred = line[0]
    cat = line[1].split('/')[-2]

    print(cat_dict[pred], cat)


    

