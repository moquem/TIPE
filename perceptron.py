import pickle
import numpy
from PIL import Image
import os


with open('poids_init','rb') as f1:
    poids = pickle.load(f1)

def calcul_res(img,poids):
    sum = 1
    for i in range(200):
        for j in range(300):
            for c in range(3):
                sum += img[i,j,c]*poids[i][j][c]
    return sum

def weight_update(poids,img,erreur):
    for i in range(200):
        for j in range(300):
            for c in range(3):
                poids[i][j][c] = poids[i][j][c] + erreur * img[i,j,c] * 0.001

for nb_img in range(321,480):
    if nb_img % 2 == 1:
        lien = 'bdd1/fire_dataset/fire_images/fire.'+str(nb_img//2)+'.png'
    else:
        lien = 'bdd1/fire_dataset/non_fire_images/non_fire.'+str(nb_img//2)+'.png'

    img = Image.open(lien)
    image = img.resize((200,300))
    img_resize = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)

    erreur = 1


    while erreur != 0:
        print("coucou")
        res = calcul_res(img_resize,poids)

        if res <= 0:
            y = 0
        else:
            y = 1

        erreur = nb_img % 2 - y
        weight_update(poids,img_resize,erreur)
    print(res,nb_img%2)
    print("une image finie yipi",nb_img)

    with open('poids_init','wb') as f1:
        pickle.dump(poids,f1)
