import csv
import os

os.chdir(r'C:\Users\imhen\PycharmProjects\45X_ML_Projects\Customized_dataset\fire_dataset')

flag = 'non'

with open('mycsv.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    for filename in os.listdir():
        if flag in filename:
            index = '1'
        else:
            index = '0'
        thewriter.writerow([filename, index])
