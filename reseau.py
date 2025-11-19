import numpy as np

class Reseau:

    def __init__(self, poids):
        self.poids = poids #np array, la i-ième liste contient les poids associés au i-ième neurone de la couche cachée
        pass

    def forwardProp(self, data):
        #renvoie résultat du réseau quand on met data en input (nombre compris entre 0 et 1)
        pass

    def descenteGradient(self, data_entrainement, nb_iterations):
        #change les poids avec l'algo du gradient
        pass

    def backProp(self, data, resultat_attendu):
        #renvoie le gradient de la fonction de cout (array 1d)
        pass

    def fonctionCout(self, resultat, resultat_attendu):
        pass



def fonctionActivation(x):
    pass

def deriveeActivation(x):
    pass
