import sys
from pathlib import Path
#import arcpy
import csv      
import os

# ws = "D:\\Project\\DWD_Projekt2_Datensaetze_local\\Daten_Satz_Coburg\\Ergebnis.gdb"
# arcpy.env.workspace = ws


# arcpy.env.overwriteOutput=True

# in_features = "gebaeude_straktur_raster_grid"
# dependent_variable = "LST"
# model_type = "continuous"
# output_features = "ErgebnisGLSRegression"
# explanatory_variables = ["hohe","flaeche","dachflaeche","volume","buildup","imprevious","grass","baeumedichte",
#                          "DGM","NDVI"]

# fields = ["OBJECTID","hohe","flaeche","dachflaeche","volume","buildup","imprevious","grass","baeumedichte","DGM",
#           "NDVI","LST","OID","Geometry","Shape_Length","Shape_Area"]

# fieldnames = ["OBJECTID","hohe","flaeche","dachflaeche","volume","buildup","imprevious","grass","baeumedichte","DGM",
#           "NDVI"]

b = Path(__file__).parents[1].resolve()
CSVFile = Path(f"{b}/data/modeldata.csv").resolve()
print(CSVFile)

# fc = "D:\\Project\\DWD_Projekt2_Datensaetze\\LoD2GridSplit\\Ergebnis.gdb\\gebaeude_straktur_raster_grid"



# arcpy.conversion.ExportTable(in_table=in_features, out_table=CSVFile, use_field_alias_as_name="NOT_USE_ALIAS",
#                              field_mapping=mapS)


# def tableToCSV(input_tbl, csv_filepath):
#     input_tbl_copy = input_tbl
#     fld_list = arcpy.ListFields(input_tbl_copy)
#     fld_names = [fld.name for fld in fld_list]
#     with open(csv_filepath, 'w') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(fld_names)
#         with arcpy.da.SearchCursor(input_tbl_copy, fld_names) as cursor:
#             for row in cursor:
#                 writer.writerow(row)
#         print(CSVFile + " CREATED")
#     csv_file.close()





# tableToCSV(in_features, CSVFile)



#
# with open(CSVFile, 'w') as f:
#     f.write(','.join(fields)+'\n') #csv headers
#     with arcpy.da.SearchCursor(in_features, fields) as cursor:
#         for row in cursor:
#             f.write(','.join([str(r) for r in row])+'\n')


