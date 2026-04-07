import pandas as pd
from pandas import read_csv

import config

#缺失比例统计
data=pd.read_csv(config.RAW_TRAIN_DATA,index_col=0)
sum=data["Survived"].count()
for i,t in zip(data.count(),data.columns):
    if i!=sum:
        print(f"{t}缺失{(sum-i)/sum*100}%的数据")

