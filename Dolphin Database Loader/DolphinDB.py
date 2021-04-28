# Utilize Dolphin Database saveText function:
login("admin","123456")
TableName.saveText("DataPath/FileName.csv",',', true)

# if the data is too large to be put in to RAM and outputed as CSV, use pipeline as well:
login("admin","123456")
v = 2015.01M..2016.12M
def queryData(m){
    return select * from loadTable("DataPath", "TableName")  # could add addtional query condition here
    # return select * from loadTable("dfs://stockDB", "stockData")
}
def saveData(tb){
    TableName.saveText("DataPath/FileName.csv",',', true)
}
pipeline(each(partial{QueryData}, v),saveData)

python scripts/dump_bin.py dump_all --csv_path ~/.qlib/csv_data/DolphinDB.csv --qlib_dir ~/.qlib/qlib_data/DolphinDB.csv --include_fields v,vw,o,c,h,l,t,n,d --date_field_name t



# what I tried out last week:
# 1. install SetupDeolphinDBForExcelRelease.msi. This is an excel add-in for dolphin database
# 2. Open an empty excel, under "Data" click "DolphinDB for Excel"
# 3. Type Server address, UserID and password
# 4. Load a table using "loadTable" function
# 5. Right click the local variables on the left and hit "Export"
# 6. Data would be outputted as PivitTable
# 7. From here it's very esay to convert a PivitTable to a csv file