# import package
import tensorflow as tf
import numpy as np 
import os 
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil 
import pandas as pd

CLASS_TYPES = ["glioma","meningioma","notumor","pituitary"]
## gg is glioma class = 0 
## m is meni class = 1
## image(i) is no tumor class =2
# p is pitu class = 3
image_size = (150,150)

def move_file(result,source_root,result_root):
    # for i, pred in enumerate(predicted_classes): 
    files = os.listdir(source_root + "/Patient")
    for i, pred in enumerate(result):
        source_file = os.path.join(source_root,"Patient",files[i])
        destination = os.path.join(result_root,CLASS_TYPES[pred],files[i])
        shutil.move(source_file,destination)
    
def create_csv_file(result_root):
    glioma = {"Name": []}
    meningioma = {"Name":[]}
    notumor = {"Name":[]}
    pituitary = {"Name": []}
    
    for CLASS in CLASS_TYPES:
        folder_path = os.path.join(result_root,CLASS)
        files = os.listdir(folder_path)
        name_list = []
        for file in files:
            name_list.append(os.path.splitext(file)[0])
        match CLASS:
            case "glioma":
                 glioma["Name"]= name_list
            case "meningioma":
                meningioma["Name"]= name_list
            case "notumor":
                notumor["Name"]= name_list
            case "pituitary":
                pituitary = name_list
    df_glioma = pd.DataFrame(glioma)
    df_meningioma = pd.DataFrame(meningioma)
    df_notumor = pd.DataFrame(notumor)
    df_pituitary = pd.DataFrame(pituitary)
    with pd.ExcelWriter('Result/Brain_Tumour_Classification.xlsx', engine='xlsxwriter') as writer: 
        df_glioma.to_excel(writer, sheet_name=CLASS_TYPES[0], index=False) 
        df_meningioma.to_excel(writer, sheet_name=CLASS_TYPES[1], index=False)
        df_notumor.to_excel(writer, sheet_name=CLASS_TYPES[2], index=False)
        df_pituitary.to_excel(writer, sheet_name=CLASS_TYPES[3], index=False)
    print("Excel file with patients' data created successfully.")
       
        
        

def model_predict(model, source_path):
    ## get dataset 
    datagen =  ImageDataGenerator(rescale=1./255)
    dataset = datagen.flow_from_directory(source_path,
                                                target_size=image_size,
                                                batch_size=32,
                                                
                                                shuffle=False)
    ## model predict
    classification_result = model.predict(dataset)
    predicted_classes = classification_result.argmax(axis=-1)
    return predicted_classes
    

def main():
    root = "Brain_Dataset"
    result_root = "Result"
    # make the directory for storing the result
    for CLASS in CLASS_TYPES:
        folder_path = os.path.join(result_root,CLASS)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Directory {CLASS} created")
    ## load model and do the model predict
    model = tf.keras.models.load_model('tumor_classification_model.keras')
    result = model_predict(model,root)
    move_file(result,root,result_root)
    create_csv_file(result_root)
    
    
            

if __name__ == '__main__':
    main()
