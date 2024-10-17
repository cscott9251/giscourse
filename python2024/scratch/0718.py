# Exercise "City Enumeration"
# 1.	Make a list of your 3 favorite cities!

# 2.	Add two more popular cities.

# 3.	Add Gelsenkirchen (can't be in the list yet) at the front of the list!

# 4.	It was obviously a mistake, delete Gelsenkirchen again.

# 5.	Count how many times Cologne appears in the list. (Ask Mr. Google how to do that...)

# 6.	Sort the list.

# 7.	Invert sorting.

# 8.	What is your favorite city's index position?

# 9.	How can you delete a value from a list without running the risk of an error message appearing because that value doesn't appear in the list?


cities = ["Berlin","Budapest","Johannesburg"]

print(cities)

cities += "Oslo","Asgardstrand"

print(cities)

cities.insert(0,"Gelsenkirchen")

print(cities)

cities.remove("Gelsenkirchen")

print(cities)

print(cities.count("Cologne"))

print(cities)

cities.sort()

print(cities)

cities.sort(reverse=True)

print(cities)

print(cities.index("Budapest"))

# if "sdsf" in cities:
#     cities.remove("sdsf")
    
    
cities += "sdsf", "sdsf", "sdsf", "sdsf"

print(cities)
    
while "sdsf" in cities: cities.remove("sdsf")

print(cities)
    