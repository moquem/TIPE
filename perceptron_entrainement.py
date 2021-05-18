"""Lance un entraînement pour une nouvelle grille de poids."""

from perceptron_fonctions import *
import pickle
import numpy
from PIL import Image
from math import *
import time
import random


poids_init()
poids = recup_poids()
entrainement_total(poids,10)
