import matplotlib.pyplot as plt
import csv
:wq




















clear
clear
:q
x = []
y = []

with open('example.txt','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                        x.append(int(row[0]))
                                y.append(int(row[1]))
