import pandas as pd
from shutil import copy

def main() -> None:
    PH2 = pd.read_excel('resources/PH2/PH2Dataset/PH2_dataset.xlsx')

    com = PH2[PH2["Unnamed: 2"] == 'X']["Unnamed: 0"]
    aty = PH2[PH2["Unnamed: 3"] == 'X']["Unnamed: 0"]
    mel = PH2[PH2["Unnamed: 4"] == 'X']["Unnamed: 0"]

    classes = [(com, 'com'), (aty, 'aty'), (mel, 'mel')]

    count = 0
    for id_list, type in classes:
        for id in id_list:
            target      = f'resources/PH2/PH2Dataset/PH2_Dataset_images/{id}/{id}_Dermoscopic_Image/{id}.bmp'
            if count % 5 == 0:
                destination = f'resources/PH2/testing_data/{type}/{id}.bmp'
            else:
                destination = f'resources/PH2/training_data/{type}/{id}.bmp'
            copy(target, destination)
            count += 1

if __name__ == '__main__':
    main()