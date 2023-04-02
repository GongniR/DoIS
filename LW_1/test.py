import pandas as pd
import numpy as np
import os

text = "C:\\Users\\gongn\Downloads\\aspects.txt"

import io

df = pd.DataFrame({'GUID':[], 'NAME':[], 'DATE': []})
with io.open(text, encoding='utf-8') as file:
    for line in file:
        if line =='Job queue status\n':
            continue
        df.loc[len(df.index )] = line.replace('\n', '').split('|')

print(df)
df.to_csv('aspects.csv',encoding='utf-8-sig' )
