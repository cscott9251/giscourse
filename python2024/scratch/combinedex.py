

JAN = [34,64,23,213,4,4,5,4,34,76,35,466,3,2000,1,7]
FEB = [55,66,3,2,3,4,5,7,44,23,67,62,3,67,785,3]

monthname = ("JAN","FEB")

print(len(JAN))
print(len(FEB))


JANFEB = [JAN,FEB]

print(JANFEB)

currentyear=0
currentmonth=0

for jan, feb in zip(JAN, FEB):
    
    print(currentyear, jan, feb)
    
    currentyear+=1
    
for month in JANFEB:
    currentmonth=-1
    currentyear += 1
    print(currentyear)
    for year in month:
            print(monthname[currentmonth])
            print(year[currentyear])


        
