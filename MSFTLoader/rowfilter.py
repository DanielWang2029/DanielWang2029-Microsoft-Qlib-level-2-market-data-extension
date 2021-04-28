import pandas as pd
data = pd.read_csv('msftdata.csv')
data = data.drop_duplicates(subset='t', keep='first')
data.to_csv("newdata.csv", index=False)
