import pickle
import numpy
import cv2
from math import *
import time
import random
from PIL import Image

def poids_init():
    """Fonction pour créer une nouvelle grille de poids."""

    print("Initialisation des poids")

    poids_act = [[[random.random(),random.random(),random.random()] for _ in range(300)] for _ in range(200)]
    l = pickle.load(open('poids','rb'))
    l.append(poids_act)
    pickle.dump(l,open('poids','wb'))

    s = pickle.load(open('stat','rb'))
    s.append([verif(poids_act)])
    pickle.dump(s,open('stat','wb'))

def recup_poids():
    """Permets de récupérer la dernière grille de poids en cours d'entraînement."""
    liste_poids = pickle.load(open('poids','rb'))
    return liste_poids[-1]

def calcul_res(img,poids):
    """Calcul le poids d'une image."""
    sum = 1
    for i in range(200):
        for j in range(300):
            for c in range(3):
                sum += (img[i,j,c]/255)*poids[i][j][c]
    return sum

def weight_update(poids,img,erreur):
    """Met à jour la grille de poids en cours d'entraînement en fonction de l'erreur de l'image."""
    for i in range(200):
        for j in range(300):
            for c in range(3):
                poids[i][j][c] = poids[i][j][c] + erreur * img[i,j,c] * 0.001
    return poids

def fc_sigmoide(res):
    """Classe l'image en fonction de son poids."""
    try:
        return 1/(1+numpy.exp(-res))
    except:
        return 0


def val_image(poids,type,num):
    """Renvoie la classe d'une image."""
    if type == 1:
        lien = 'dataset/fire/fire_verification/'+str(num)+'.png'
    else:
        lien = 'dataset/non_fire/non_fire_verification/'+str(num)+'.png'

    img = cv2.imread(lien)

    res = calcul_res(img,poids)
    val = fc_sigmoide(res)

    return val


def entrainement(poids):
    """Entraîne une grille de poids à l'aide de toutes les fonctions définies plus tôt, pour une image."""
    type = random.randint(0,1)
    if type == 1:
        num = random.randint(0,18577)
        lien = 'dataset/fire/fire_entrainement/'+str(num)+'.png'
    else:
        num = random.randint(0,13757)
        lien = 'dataset/non_fire/non_fire_entrainement/'+str(num)+'.png'

    img = cv2.imread(lien)

    erreur = 1

    while abs(erreur) > 0.1:
        res = calcul_res(img,poids)
        val = fc_sigmoide(res)
        erreur = type - val

        poids = weight_update(poids,img,erreur)

        erreur = type - val

    return poids

def session(nb_image,poids):
    """Lance une session d'entraînement."""
    for i in range(nb_image):
        print("train=",i)
        poids = entrainement(poids)

    return poids

def entrainement_total(nb_session,nb_image):
    """lance un gros entrainement, enregistre le temps utilisé, les statistiques etc"""

    temps_deb = time.time()
    p = pickle.load(open('poids','rb'))
    s = pickle.load(open('stat','rb'))

    poids = p[-1]
    stat_act = s[-1]

    for i_sess in range(nb_session):
        poids = session(nb_image,poids)
        print("fin session :",i_sess+1)
        result = verif(poids)
        stat_act.append(result)


        p[-1] = poids
        s[-1] = stat_act

        pickle.dump(p,open('poids','wb'))
        pickle.dump(s,open('stat','wb'))
        
    temps_fin = time.time()

    print(temps_fin - temps_deb)

    return



def verif(poids_act):
    """Lance une session de vérification pour voir le pourcentage d'images correctement classées."""
    juste = 0
    faux_pos = 0
    faux_neg = 0

    for i in range(500):

        print('verif=',i)

        type = random.randint(0,1)

        if type == 1:
            num = random.randint(0,6418)
            val = val_image(poids_act,type,num)
            if val >= 0.5:
                juste += 1
            else:
                faux_neg += 1

        else:
            num = random.randint(0,3869)
            val = val_image(poids_act,type,num)
            if val < 0.5:
                juste += 1
            else:
                faux_pos += 1


    result = (juste/5,faux_pos/5,faux_neg/5)
    print("accuracy,faux_pos,faux_neg:",result)
    return result

def classification(poids,lien):
    """Donne le pourcentage de classification d'une image."""
    pic = Image.open(lien)
    img = numpy.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)

    res = calcul_res(img,poids)
    val = fc_sigmoide(res)

    print("fire",val,"non_fire",1-val)
