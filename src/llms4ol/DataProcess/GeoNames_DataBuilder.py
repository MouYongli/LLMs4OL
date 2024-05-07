from pathlib import Path
import pandas as pd
import numpy as np

def convert_csv(path: Path, names: list = None, sep: str=',', low_memory: bool= True, header: list=None) -> pd:
    return pd.read_csv(path, sep=sep, header=header, low_memory=low_memory, names=names)

def write_csv(data: pd, path: Path):
    data.to_csv(path, index=False)

def load_df(path: Path, columns: list=None) -> pd:
    if columns is None:
        data_frame = pd.read_csv(path)
    else:
        data_frame = pd.read_csv(path, names=columns)
    
    return data_frame

def extract_featureCodes_to_csv():
    raw_file = "../../assets/Datasets/SubTaskB.1-GeoNames/Additional_Data/featureCodes_en.txt"
    processed_file = "../../assets/Datasets/SubTaskB.1-GeoNames/Additional_Data/featureCodes_en.csv"
    geo_df = convert_csv(raw_file,
                                 sep='\t', names=["Code", "Name", "Definition"])
    l1_mapper = {
        "A": ["country", "state", "region"], "H": ["stream", "lake"], "L": ["parks", "area"],
        "P": ["city", "village"], "R": ["road", "railroad"], "S": ["spot", "building", "farm"],
        "T": ["mountain", "hill", "rock"], "U": ["undersea"], "V": ["forest", "heath"]
    }
    geo_df['L1'] = geo_df['Code'].apply(lambda X: X[0])
    geo_df['L1Name'] = geo_df['L1'].apply(lambda X: l1_mapper[X])
    geo_df['L2'] = geo_df['Code'].apply(lambda X: X[2:])
    geo_df['L2Name'] = geo_df['Name']
    geo_df = geo_df.drop(['Name'], axis=1)
    print(geo_df[:10])
    write_csv(geo_df, processed_file)

#http://www.geonames.org/export/codes.html
def extract_geoType_to_array():
    all_types = []
    csv_file = "../../assets/Datasets/SubTaskB.1-GeoNames/Additional_Data/featureCodes_en.csv"
    df = load_df(csv_file)
    name_mappers = {
        "A": "country,state,region",
        "H": "stream,lake",
        "L": "parks,areas",
        "P": "city,village",
        "R": "road,railroad",
        "S": "spot,building,farm",
        "T": "mountain,hill,rock",
        "U": "undersea",
        "V": "forest,heath"
    }
    a_list = df['L1'].tolist() # super class
    b_list = df['L2Name'].tolist()# sub class
    parent = []
    child = []
    for text_a, text_b in zip(a_list, b_list):
        parent.append(name_mappers[text_a])
        child.append(text_b.lower())
    unique_supertypes = list(set(parent))
    unique_subtypes = list(set(child))
    print(f"There are {len(unique_subtypes)} sub-types and {len(unique_supertypes)} super-types in total.")
    return parent,child

extract_geoType_to_array()