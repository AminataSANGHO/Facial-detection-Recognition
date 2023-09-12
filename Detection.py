from PIL import Image, ImageDraw
import face_recognition
import cv2
import os

def findEncodings(images):
    encodeList = []
    for img in images:
        if(img is not None):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            encoded_face = face_recognition.face_encodings(img)[0]
            encodeList.append(encoded_face)
    return encodeList

def detecter_visage(pathImagesTest):
    images = []
    mylist = os.listdir(pathImagesTest)
    for cl in mylist:
        curImg = cv2.imread(f'{pathImagesTest}/{cl}')
        images.append(curImg)
    
    emp_visages = face_recognition.face_locations(images)
    for emp_visage in emp_visages:
        gauche, haut, droite, bas = emp_visage
        print(f"Visage détecté aux coordonnées : Gauche={gauche}, Haut={haut}, Droite={droite}, Bas={bas}")
        # Dessinez une boite autour du visage à l'aide du module Pillow
        image_pil = Image.open(curImg)
        draw = ImageDraw.Draw(image_pil)
        draw.rectangle(((gauche, haut), (droite, bas)), outline=(255, 0, 255))
        image_pil.show()


pathImagesTest = './imagesTest/'
detecter_visage(pathImagesTest)
