from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import os
import cv2

def findEncodings(images):
    encodeList = []
    for img in images:
        if(img is not None):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            encoded_face = face_recognition.face_encodings(img)[0]
            encodeList.append(encoded_face)
    return encodeList

def reconnaitre_visage(pathImageaTester, pathImagesTest):
    images = []
    classNames = []
    mylist = os.listdir(pathImagesTest)
    ImageaTester = os.listdir(pathImageaTester)
    for cl in mylist:
        curImg = cv2.imread(f'{pathImagesTest}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    encodage_visage_connu = findEncodings(images)
    
    for cl in ImageaTester:
        
        image_inconnu = face_recognition.load_image_file(f'{pathImageaTester}/{cl}')

        # Trouver tous les visages et encodages de visage dans l'image inconnue
        emp_visage_inconnu = face_recognition.face_locations(image_inconnu)
        encodage_visage_inconnu = face_recognition.face_encodings(image_inconnu, emp_visage_inconnu)
       
        image_pil = Image.fromarray(image_inconnu)
        draw = ImageDraw.Draw(image_pil)
        
        # Traverser chaque visage trouvé dans l'image inconnue
        for (haut, droite, bas, gauche), encodage_visage in zip(emp_visage_inconnu, encodage_visage_inconnu):
            # Voir si le visage correspond au visage connu
            corresp = face_recognition.compare_faces(encodage_visage_connu, encodage_visage)
            # [True, False]

            nom = "Inconnu"

            # Ou à la place, utilisez le visage connu avec la plus petite distance par rapport au nouveau visage
            distances_visages = face_recognition.face_distance(encodage_visage_connu, encodage_visage)
            meilleur_indice = np.argmin(distances_visages)
            if corresp[meilleur_indice]:
                nom = classNames[meilleur_indice]

            # Dessinez une boîte autour du visage à l'aide du module Pillow
            draw.rectangle(((gauche, haut), (droite, bas)), outline=(255, 0, 255))

            # Dessinez une étiquette avec un nom sous le visage
            largeur_texte, hauteur_texte = draw.textsize(nom)
            draw.text((gauche + 6, bas - hauteur_texte - 5), nom, fill=(255, 255, 255, 255))

        image_pil.show()

pathImageaTester = "./imageaTester"
pathImagesTest = "./imagesTest"
reconnaitre_visage(pathImageaTester, pathImagesTest)
