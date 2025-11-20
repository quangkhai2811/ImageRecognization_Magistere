import numpy as np

class Reseau:

    def __init__(self, tailles_couches, fonctionActivation, poids_initiaux):
        
        self.tailles_couches = tailles_couches #liste avec la taille de chaque couche
        self.nb_couches = len(self.tailles_couches)
        self.fonctionActivation = fonctionActivation
        self.poids = poids_initiaux #liste de matrices np, dont la taille est nb_couches-1
        self.activations = [np.ones(taille+1) for taille in self.tailles_couches]


    def forwardProp(self, data):
        self.activations[0][:-1] = data
        for i in range(self.nb_couches-1):
            self.activations[i+1][:-1] = self.fonctionActivation(np.dot(self.poids[i+1], self.activations[i]))


    def descenteGradient(self, data_entrainement, nb_iterations):
        #change les poids avec l'algo du gradient
        pass


    def backProp(self, data, resultat_attendu):
        #renvoie le gradient de la fonction de cout (array 1d)
        pass


    def fonctionCout(self, resultat, resultat_attendu):
        pass
