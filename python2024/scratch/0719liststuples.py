import random

baumarten = ("Yree 1", "Free 2", "Tree 3", "Aree 4", "Mree 5")

# def sort_tup(tup):
#     tup.sort(key= lambda x: x[0])



for art in range(len(baumarten)):
    
    print(baumarten[art])
    
print("----------------------------------------------")    
    
baumarten_list = list(baumarten)
baumarten_list.sort()

for tree in baumarten_list:
    print(tree)

print("----------------------------------------------")
    
print(baumarten[random.randint(0,len(baumarten)-1)])

print("----------------------------------------------")

for n in range(10000):
    print(n, baumarten[random.randint(0,len(baumarten)-1)])