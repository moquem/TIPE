import pickle
import numpy
import cv2
from math import *
import time
import random
from PIL import Image

def poids_init():
    """Fonction pour créer une nouvelle grille de poids."""

    poids_act = [[[random.random(),random.random(),random.random()] for _ in range(300)] for _ in range(200)]
    l = pickle(open('poids','rb'))
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
    """Entraîne une grille de poids à l'aide de toutes les fonctions définies plus tôt."""
    type = random.randint(0,1)
    if type == 1:
        num = random.randint(0,2070)
        lien = 'dataset/fire/fire_entrainement/'+str(num)+'.png'
    else:
        num = random.randint(0,3552)
        lien = 'dataset/non_fire/non_fire_entrainement/'+str(num)+'.png'

    img = cv2.imread(lien)

    erreur = 1

    while abs(erreur) > 0.1:
        res = calcul_res(img,poids)
        val = fc_sigmoide(res)
        erreur = type - val

        poids = weight_update(poids,img,erreur)

        erreur = type - val

    print("type",type)
    print(val,1-val)

    return poids

def session(nb_image,poids):
    """Lance une session d'entraînement."""
    temps_deb = time.time()
    for i in range(nb_image):
        poids = entrainement(poids)
    temps_fin = time.time()

    return poids

def entrainement_tot(nb_session,nb_image):
    """lance un gros entrainement, enregistre le temps utilisé, les statistiques etc"""

    temps_deb = time.time()
    p = pickle.load(open('poids','rb'))
    s = pickle.load(open('stat','rb'))

    poids = p[-1]
    stat_act = s[-1]

    for i_sess in range(nb_session):
        poids = session(nb_image,poids)
        stat_act.append(verif(poids))

    p[-1] = poids
    s[-1] = stat_act

    pickle.dump(p,open('poids','wb'))
    pickle.dump(s,open('stat','wb'))



def verif(poids_act):
    """Lance une session de vérification pour voir le pourcentage d'images correctement classées."""
    juste = 0
    faux_pos = 0
    faux_neg = 0

    for i in range(100):
        print(i)
        type = random.randint(0,1)

        if type == 1:
            num = random.randint(0,1000)
            val = val_image(poids_act,type,num)
            if val >= 0.5:
                juste += 1
            else:
                faux_neg += 1

        else:
            num = random.randint(0,1500)
            val = val_image(poids_act,type,num)
            if val < 0.5:
                juste += 1
            else:
                faux_pos += 1


    result = (juste/10,faux_pos,faux_neg)

    with open('result_entrainement','rb') as f1:
        result_ent = pickle.load(f1)

    result_ent[-1].append(result)
    with open('result_entrainement','wb') as f1:
        pickle.dump(result_ent,f1)

    return None

def classification(poids,lien):
    """Donne le pourcentage de classification d'une image."""
    pic = Image.open(lien)
    img = numpy.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)

    res = calcul_res(img,poids)
    val = fc_sigmoide(res)

    print("fire",val,"non_fire",1-val)
