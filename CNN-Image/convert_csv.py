import csv

with open('prediction_6.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    headings = next(spamreader)
    print(headings)
    headings = ["id", "class"]
    print(headings)
    rows = []
    for row in spamreader:
        row = list(map(float, row))
        new_row = [None, None]
        new_row[0] = int(row[0]) + 1
        new_row[1] = row.index(max(row[1:]))
        rows.append(new_row)
        row = new_row
    
with open("prediction_new.csv", 'w', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headings)
    f_csv.writerows(rows)

