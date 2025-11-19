from reseau import Reseau
import numpy as np

def poidsInitiaux():
    #renvoie une liste de poids générés aléatoirement
    pass

def importData():
    #renvoie deux arrays : un pour les données d'entrainement et l'autre pour les données de test
    pass

#création du réseau
poids = poidsInitiaux()
reseau = Reseau(poids)

#phase d'entrainement
data_entrainement, data_test = importData()
nb_iterations = 10
reseau.descenteGradient(data_entrainement, nb_iterations)

#phase de test
