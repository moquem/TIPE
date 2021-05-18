"""Permets de classer une photo donnée avec le dernier perceptron entraîné."""


from perceptron_fonctions import *
import pickle
import numpy
from PIL import Image
from math import *
import time
import random

poids_act = recup_poids()
lien = input()
classification(poids_act,lien)
