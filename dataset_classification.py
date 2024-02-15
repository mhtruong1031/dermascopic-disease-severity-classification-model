import pandas as pd
from shutil import copy

PH2 = pd.read_excel('resources/PH2/PH2Dataset/PH2_dataset.xlsx')

com = PH2[PH2["Unnamed: 2"] == 'X']["Unnamed: 0"]
aty = PH2[PH2["Unnamed: 3"] == 'X']["Unnamed: 0"]
mel = PH2[PH2["Unnamed: 4"] == 'X']["Unnamed: 0"]

classes = [(com, 'com'), (aty, 'aty'), (mel, 'mel')]

for id_list, type in classes:
    for id in id_list:
        target      = f'resources/PH2/PH2Dataset/PH2_Dataset_images/{id}/{id}_Dermoscopic_Image/{id}.bmp'
        destination = f'resources/PH2/Classified_Dataset/{type}/{id}.bmp'
        copy(target, destination)