import pandas as pd
import numpy as np

path = "C:\\Users\\User\\Desktop\\datialessio.csv"

data = pd.read_csv(path, encoding='latin-1')

to_keep = ['Shipment Date', 'Quantity', 'Product Tax Code',
           'OUR_PRICE Tax Inclusive Selling Price', 'OUR_PRICE Tax Amount',
           'OUR_PRICE Tax Exclusive Selling Price', 'SHIPPING Tax Inclusive Selling Price', 'SHIPPING Tax Amount',
       'SHIPPING Tax Exclusive Selling Price',
       'SHIPPING Tax Inclusive Promo Amount', 'SHIPPING Tax Amount Promo',
       'SHIPPING Tax Exclusive Promo Amount', 'Seller Tax Registration', 'VAT Invoice Number']

my_col = data[to_keep]


iva_map = {"A_GEN_STANDARD": 22, "A_OUTDOOR_FERTILIZER": 4, "A_OUTDOOR_SEEDS": 10}

final = ['VAT Invoice Number', "Shipment Date", "ASIN", "SKU",
         'OUR_PRICE Tax Inclusive Selling Price', 'Product Tax Code', 'Invoice Url']
real = data[final]

# for x in my_col.groupby('Product Tax Code'):
#     print(x[0])
#     print(x[1])

all_id = list()
A_GEN_STANDARD = list()
A_OUTDOOR_FERTILIZER = list()
A_OUTDOOR_SEEDS = list()

for x in real.iterrows():
    print(x[1])
    value_id = x[1]['VAT Invoice Number']
    all_id.append(int(value_id.split("-")[-1]) if type(value_id) == str  else 0)
    if x[1]['Product Tax Code'] == "A_GEN_STANDARD":
        A_GEN_STANDARD.append(x[1]['OUR_PRICE Tax Inclusive Selling Price'])
        A_OUTDOOR_FERTILIZER.append(0)
        A_OUTDOOR_SEEDS.append(0)
    elif x[1]['Product Tax Code'] == "A_OUTDOOR_FERTILIZER":
        A_GEN_STANDARD.append(0)
        A_OUTDOOR_FERTILIZER.append(x[1]['OUR_PRICE Tax Inclusive Selling Price'])
        A_OUTDOOR_SEEDS.append(0)
    else:
        A_GEN_STANDARD.append(0)
        A_OUTDOOR_FERTILIZER.append(0)
        A_OUTDOOR_SEEDS.append(x[1]['OUR_PRICE Tax Inclusive Selling Price'])

real["id"] = all_id
real["A_GEN_STANDARD"] = A_GEN_STANDARD
real["A_OUTDOOR_FERTILIZER"] = A_OUTDOOR_FERTILIZER
real["A_OUTDOOR_SEEDS"] = A_OUTDOOR_SEEDS



fin = real.sort_values("id")
fin.to_csv("C:\\Users\\User\\Desktop\\prova.csv")




