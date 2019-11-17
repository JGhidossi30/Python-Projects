import pickle
from xlrd import open_workbook

book = open_workbook("Data-Clean.xlsx")
sheet = book.sheet_by_index(0)

input_data = []
output_data = []

for col in range(1, 37):
    feature_values = []
    for row in range(1, 12):
        feature_values.append(sheet.cell(row, col).value)
    input_data.append(feature_values)

for col in range(1, 37):
    output_data.append([sheet.cell(12, col).value])

print (output_data)

pickle.dump(input_data, open("input.pkl", "wb"))
pickle.dump(output_data, open("output.pkl", "wb"))

#print (pickle.load(open("input.pkl", "rb")))
#print (pickle.load(open("output.pkl", "rb")))

