import csv 
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('GTK3Agg')

with open('out_out_out.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))

plt.plot(x,y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


