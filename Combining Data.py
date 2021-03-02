import glob
import pandas as pd
import os

os.chdir("./Extracted Data")
extension='csv'
all_filenames=[i for i in glob.glob(f'*.{extension}')]

combined_data=pd.concat([pd.read_csv(f) for f in all_filenames])
combined_data.to_csv("../combined_data.csv",index=False,encoding='utf-8-sig')