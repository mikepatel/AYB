"""
Michael Patel
June 2020

Project description:

File description:
"""
################################################################################
# Imports
import os
import pandas as pd


################################################################################
# Pale_Skin = 1
# .jpg
# moved to different directory, then manually triage for just black

file = "C:\\Users\\micha\\PycharmProjects\\Make-Money\\data\\list_attr_celeba.csv"

df = pd.read_csv(file)
print(df["Black"][1680:1695])


011899.jpg

