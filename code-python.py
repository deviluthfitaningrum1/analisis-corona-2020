import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

date=[]
for feb in range (15, 29+1) :
    date.append(str(feb)+ " Feb")
for mar in range (1, 26+1) :
    date.append(str(mar)+ " Mar")
print(str(len(date)))

day = []
for i in range(0, len(date)) :
    day.append(i)
print(str(len(day)))


x = np.array((1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41))

#input data active cases atau total death
y = np.array((12,12,12,12,12,10,29,29,28,48,51,54,54,57,60,65,85,106,138,200,289,401,504,663,949,1248,1581,2126,2664,3484,4434,6127,8940,13464,18965,23720,33000,42930,53697,66790,82272))
print(str(len(y)))

tabel = pd.DataFrame({
    'Day' : day,
    'Date' : date,
    'Active Cases/Total Death' : y,
})
tabel

ax = plt.plot(x,y,'o')
plt.grid()
plt.title("Plot Data Active Cases/Total Death (Nama negara)")

#Regresi Eksponensial
l=np.log(y)
m=x

n=len(y)
ml = l*m
mm = m**2

B=(n*ml.sum()-m.sum()*l.sum())/((n*mm.sum())-(m.sum())**2)
A=(l.mean()-B*m.mean())

a = np.e**A 
b = B

print("Persamaan regresi eksponensial adalah y = {:.4f}e^({:.4f}x)".format(a,b))

yDt = (l-l.mean())**2
yD = (l-A-B*m)**2

Dt = yDt.sum()
D = yD.sum()

r = np.sqrt((Dt-D)/Dt)

print("Nilai koefisien korelasinya adalah {:.4f}".format(r))

yreg=a*(np.e**(b*x))
ax = plt.plot (x,y,'o')
plt.plot(x,yreg,'r')
plt.grid()
plt.title("Kurva regresi eksponensial pertumbuhan Active Cases/Total Death (Nama negara)")
print("Persamaan regresi adalah {:.4f}e^({:.4f}x)".format(a,b))
print("Nilai koefisien korelasinya adalah {:.4f}".format(r))

#Regresi Kuadrat
m=np.log10(x)
l=np.log10(y)
n=len(x)
ml = m*l
mm = m**2
B=(n*ml.sum()-m.sum()*l.sum())/((n*mm.sum())-(m.sum())**2)
A=(l.mean()-B*m.mean())
b=B
a=10**A

print("Persamaan regresi kuadrat adalah {:.4f}x^{:.4f}".format(a,b))

yDt = (l-l.mean())**2
yD = (l-A-B*m)**2
Dt = yDt.sum()
D = yD.sum()
r = np.sqrt((Dt-D)/Dt)
print("Nilai koefisien korelasinya adalah {:.4f}".format(r))

ax = plt.plot(x,y,'o')
yreg = a*x**b
plt.plot(x,yreg,'r')
plt.grid()
print("Persamaan regresi adalah {:.4f}x^{:.4f}".format(a,b))
print("Nilai koefisien korelasinya adalah {:.4f}".format(r))
plt.title("Kurva regresi kuadrat Active Cases/Total Death (Nama negara)")

#Regresi Polinomial

from numpy import *
from scipy.interpolate import *
p3 = polyfit(x,y,3)
p4 = polyfit(x,y,4)
p5 = polyfit(x,y,5)
p6 = polyfit(x,y,6)
p7 = polyfit(x,y,7)
p8 = polyfit(x,y,8)
p9 = polyfit(x,y,9)

from matplotlib.pyplot import *
%matplotlib inline
yfit3 = p3[0]*(x**3) + p3[1]*(x**2)+ p3[2]*x+p3[3]
yfit4 = p4[0]*(x**4) + p4[1]*(x**3)+ p4[2]*x**2+p4[3]*x+p4[4]
yfit5 = p5[0]*(x**5) + p5[1]*(x**4)+ p5[2]*x**3+p5[3]*x**2+p5[4]*x+p5[5]
yfit6 = p6[0]*(x**6) + p6[1]*(x**5)+ p6[2]*x**4+p6[3]*x**3+p6[4]*x**2+p6[5]*x+p6[6]
yfit7 = p7[0]*(x**7) + p7[1]*(x**6)+ p7[2]*x**5+p7[3]*x**4+p7[4]*x**3+p7[5]*x**2+p7[6]*x+p7[7]
yfit8 = p8[0]*(x**8) + p8[1]*(x**7)+ p8[2]*x**6+p8[3]*x**5+p8[4]*x**4+p8[5]*x**3+p8[6]*x**2+p8[7]*x+p8[8]
yfit9 = p9[0]*(x**9) + p9[1]*(x**8)+ p9[2]*x**7+p9[3]*x**6+p9[4]*x**5+p9[5]*x**4+p9[6]*x**3+p9[7]*x**2+p9[8]*x+p9[9]

SStotal= len(y)*var(y)

yresid3=y-yfit3
SSresid3= sum(pow(yresid3,2))
rr3=1-SSresid3/SStotal
r3=np.sqrt(abs(rr3))
print("Koefisien korelasi orde 3 = {:.4f}".format(r3))

yresid4=y-yfit4
SSresid4= sum(pow(yresid4,2))
rr4=1-SSresid4/SStotal
r4=np.sqrt(abs(rr4))
print("Koefisien korelasi orde 4 = {:.4f}".format(r4))

yresid5=y-yfit5
SSresid5= sum(pow(yresid5,2))
rr5=1-SSresid5/SStotal
r5=np.sqrt(abs(rr5))
print("Koefisien korelasi orde 5 = {:.4f}".format(r5))

yresid6=y-yfit6
SSresid6= sum(pow(yresid6,2))
rr6=1-SSresid6/SStotal
r6=np.sqrt(abs(rr6))
print("Koefisien korelasi orde 6 = {:.4f}".format(r6))

yresid7=y-yfit7
SSresid7= sum(pow(yresid7,2))
rr7=1-SSresid7/SStotal
r7=np.sqrt(abs(rr7))
print("Koefisien korelasi orde 7 = {:.4f}".format(r7))

yresid8=y-yfit8
SSresid8= sum(pow(yresid8,2))
rr8=1-SSresid8/SStotal
r8=np.sqrt(abs(rr8))
print("Koefisien korelasi orde 8 = {:.4f}".format(r8))

yresid9=y-yfit9
SSresid9= sum(pow(yresid9,2))
rr9=1-SSresid9/SStotal
r9=np.sqrt(abs(rr9))
print("Koefisien korelasi orde 9 = {:.4f}".format(r9))

#print("Regresi Polinomial orde 5 dipilih karena memiliki koefisien korelasi paling mendekati 1")

ax = plt.plot(x,y,'o')
plot(x,polyval(p5,x), 'r-')
plt.grid()
plt.title("Kurva regresi polinomial pertumbuhan Active Cases/Total Death (Nama negara)")
print("Persamaan polinomial orde 5 adalah y = {:.4f}x^5 + {:.4f}x^4 + {:.4f}x^3 + {:.4f}x^2 + {:.4f}x + {:.4f}".format(p5[0],p5[1],p5[2],p5[3],p5[4],p5[5]))
print("Koefisien korelasinya adalah {:.4f}" .format(r5))