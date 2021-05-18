import pickle
import numpy
from PIL import Image


with open('poids_init','rb') as f1:
    poids = pickle.load(f1)

def calcul_res(img,poids):
    sum = 0
    for i in range(200):
        for j in range(300):
            for c in range(3):
                sum += img[i,j,c]*poids[i][j][c]
    return sum

print("quel numero")
numero = int(input())
print("quelle catégorie")
categorie = input()

if categorie == 'fire':
    lien = 'bdd1/fire_dataset/fire_images/fire.'+str(numero)+'.png'
else:
    lien =  'bdd1/fire_dataset/non_fire_images/non_fire.'+str(numero)+'.png'

img = Image.open(lien)
image = img.resize((200,300))
img_resize = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)

res = calcul_res(img_resize,poids)

if res > 0:
    print("c'est du feu !")
else:
    print("no. it's not fire")
