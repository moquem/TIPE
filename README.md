### Présentation

J'ai développé ce projet en CPGE (classes préparatoires aux grandes écoles) dans le cadre de mon TIPE : __Réseaux de neurones et traitement d'images appliqués à la détection d'incendie.__

Pour l'épreuve de TIPE, les perceptrons et réseaux ont été entraînés sur des bases de données de quelques milliers d'images divisées en deux catégories : des images de feu et des images sans feu.

Les codes peuvent être repris et entraînés pour n'importe quel problème de classification à deux catégories.

### Dépendance

Pour exécuter les différents programmes, il faut avoir installé __Python3__ ainsi que les modules:
- pickle
- numpy
- cv2
- PIL

### Exécution

Avant l'exécution de toutes fonctions ou programmes, il faut exécuter `initialisation.py`.
Cela va créer deux fichiers `pickle`: `poids` et `stat`, deux listes qui vont contenir les perceptrons entraînés (sous forme de grille de poids) et les statistiques pour chaque perceptron (liste contenant la précision, le nombre de faux_positif et le nombre de faux_negatif sous forme de tuple, calculés au terme de chaque session).

- `perceptron_fonctions.py` contient toutes les fonctions nécessaires à l'entraînement d'un perceptron.
Pour plus d'information sur une fonction donnée, importer `perceptron_fonctions` dans un exécuteur __Python3__, puis taper `help(nom_fonction)`.

Attention à bien modifier les chemins des images d'entraînement/vérification dans les fonctions `val_image` et `entrainement` avant de les utiliser.

- `perceptron_entrainement.py` permet de lancer un entraînement (de 10 sessions de 1000 images) sur un nouveau perceptron.

- `perceptron_eval.py` classe une image dans une des deux catégories à l'aide du dernier perceptron.

### Contraintes

Les images d'entraînement et de vérification doivent être au format 300x200.
