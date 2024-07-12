# x = 6.24

# print(round(x/3.2))

def DMSToDecimalDegrees(dd):
    dd = round(float(dd[0]) + float(dd[1])/60 + float(dd[2])/(60*60),5)
    dd=str(dd)
 # put the contents of the function here.
    return dd

TheCoordinate="38 43 50.46,9 9 17.22"

TheElements=TheCoordinate.split(",")
TheLatitude=TheElements[0]
TheLongitude=TheElements[1]
print("TheLatitude="+TheLatitude)
print("TheLongitude="+TheLongitude)

TheLatitude=(TheLatitude.split(sep=" "))
TheLongitude=(TheLongitude.split(sep=" "))

print(len(TheLatitude))
print(type(TheLatitude))
print(len(TheLongitude))
print(type(TheLongitude))

TheLatitudeValue=DMSToDecimalDegrees(TheLatitude)
print("TheLatitudeValue="+TheLatitudeValue)

TheLongitudeValue=DMSToDecimalDegrees(TheLongitude)
print("TheLongitudeValue="+TheLongitudeValue)

decimalCoord=TheLatitudeValue+","+TheLongitudeValue

print(decimalCoord)
