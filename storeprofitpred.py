import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#decision tree 
dataset = pd.read_csv('wallmarts.csv')

X = dataset.iloc[:,[0,1,4]]
y = dataset.iloc[:,3].values




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X.iloc[:,2] = labelencoder.fit_transform(X.iloc[:,2])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

y_test = sc_y.fit_transform(y_test.reshape(-1,1))
'''

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,y_train)
print((regressor.score(X_test,y_test))*100)


ypred = regressor.predict(X)
plt.plot(X_test.iloc[:,[2]],y_test)
plt.plot(X_test.iloc[:,[2]],ypred)
#for prediction 

Xnew = dataset.iloc[:,[0,1,2,3]]
ynew = dataset.iloc[:, 4]

X1 = Xnew.values
y1 = ynew.values
from sklearn.model_selection import train_test_split
X_trainn, X_testn, y_trainn, y_testn = train_test_split(X1, y1, test_size = 0.2, random_state = 0)




# Sales os store 1 dept 1 in 2010
def tot_sale(store,dept):
    sales1= 0
    sales2= 0
    sales3= 0
    sales4= 0
    sales5= 0
    sales6= 0
    for row in Xnew.values   :
        date = row[2]
        if row[0] == store and row[1] == dept and date[3] == '0' and int(date[6]) <=6:
            sales1 = sales1 + row[3]
        if row[0] == store and row[1] == dept and date[3] == '0' and int(date[6]) >6:
            sales2 = sales2 + row[3]
        if row[0] == store and row[1] == dept and date[3] == '1' and int(date[6]) <=6:
            sales3 = sales3 + row[3]
        if row[0] == store and row[1] == dept and date[3] == '1' and int(date[6]) >6:
            sales4 = sales4 + row[3]
        if row[0] == store and row[1] == dept and date[3] == '2' and int(date[6]) <=6:
            sales5 = sales5 + row[3]
        if row[0] == store and row[1] == dept and date[3] == '2' and int(date[6]) >6:
            sales6 = sales6 + row[3]
    sale = [sales1,sales2,sales3,sales4,sales5,sales6]
    return sale

years = [1,2,3,4,5,6]
sale1 = tot_sale(11,10)


plt.plot([1,2,3,4,5,6],sale1)
plt.scatter([1,2,3,4,5,6],sale1)

d = {"years":years,"sales":sale1}
df = pd.DataFrame(d, columns=['years','sales'])


a1 = df.iloc[[0,2,4],[-2]].values
a2 = df.iloc[[0,2,4],1].values.astype(int)



import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow
class mywin(QMainWindow):
    def __init__(self):
        super(mywin,self).__init__()
        self.setGeometry(100,100,600,600)
        self.setWindowTitle("Enter the inputs")
        self.initUI()

    def initUI(self):
        self.l1=QtWidgets.QLabel(self)
        self.l1.setText("Enter the Store Number")
        self.l1.adjustSize()
        self.l1.move(50,30)
        self.t1=QtWidgets.QLineEdit(self)
        self.t1.move(220,30)
        self.t1.resize(220,20)
       
        self.l2=QtWidgets.QLabel(self)
        self.l2.setText("Enter the Dept")
        self.l2.move(50,50)
        self.t2=QtWidgets.QLineEdit(self)
        self.t2.move(220,55)
        self.t2.resize(220,20)
       
   
        self.l3=QtWidgets.QLabel(self)
        self.l3.setText("Enter the Date(D/M/Y)")
        self.l3.move(50,80)
        self.l3.adjustSize()
        self.t3=QtWidgets.QLineEdit(self)
        self.t3.move(220,80)
        self.t3.resize(30,20)
        self.t4=QtWidgets.QLineEdit(self)
        self.t4.move(260,80)
        self.t4.resize(30,20)
        self.t5=QtWidgets.QLineEdit(self)
        self.t5.move(300,80)
        self.t5.resize(30,20)
   
   
        self.b1=QtWidgets.QPushButton(self)
        self.b1.move(150,130)
        self.b1.setText("OK")
        self.b1.clicked.connect(self.cl)
       
    def cl(self):
        storeinp=self.t1.text()
        print(storeinp)
        deptinp=self.t2.text()
        print(deptinp)
        dateinp=(str(self.t3.text()+"-"+self.t4.text()+"-"+self.t5.text()))
        print(dateinp)
        
        
        
        
        
def win():
    app=QApplication(sys.argv)
    w=mywin()
    w.show()
    sys.exit(app.exec_())
   
win()




