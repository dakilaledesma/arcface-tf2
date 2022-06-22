from glob import glob
import shutil
import os

for fl in glob("data/orchidaceae_train/*"):
    shutil.move(fl, f"{fl}_")
    
renamed = {}
for idx, fl in enumerate(glob("data/orchidaceae_train/*")):
    bn = os.path.basename(fl)
    renamed[bn[:-1]] = idx
    shutil.move(fl, f"{fl.replace(bn, '')}{idx}")

out_str = '\n'.join([f"{k},{v}" for k, v in renamed.items()])
out_file = open("rename_results.csv", 'w')
out_file.write(out_str)
out_file.close()