# 359 Group 2 Project

## MSFT Data Loading
Attempting to load in the MSFT data without making any changes results in an error.

Running the following will edit the data and make it compatible with qlib:
```bash
import pandas as pd
data = pd.read_csv('MSFT_2004-10-31_2020-06-12_minute.csv')
data = data.drop_duplicates(subset='t', keep='first')
data.to_csv("MSFT.csv", index=False)
```
To load in the MSFT data, place the resulting MSFT.csv from above into the csv_data folder and run the following:
```bash
python scripts/dump_bin.py dump_all --csv_path  ~/.qlib/csv_data/MSFT.csv --qlib_dir ~/.qlib/qlib_data/MSFT.csv --include_fields v,vw,o,c,h,l,t,n,d --date_field_name t 
```
