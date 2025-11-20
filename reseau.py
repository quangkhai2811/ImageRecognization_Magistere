import numpy as np

class Reseau:

    def __init__(self, poids, fctActivation):
        self.poids = poids #np array, la i-ième liste contient les poids associés au i-ième neurone de la couche cachée

        pass

    def forwardProp(self, data):
        #renvoie résultat du réseau quand on met data en input (nombre compris entre 0 et 1)
        activations = list()
        # list indice = 0 est celle qui contient les entrées
        activations.append(data)

        for l in range(1, self.nb_couches):
            # get previous activations
            A_prev = activations[l-1]

            # compute pre-activation
            Z = np.dot(A_prev, self.poids[l])

            # apply pre-activation
            A = self.fctActivation(Z)

            # store values for later use
            activations[l] = A

        # compute output layer (layer L)
        A_prev = activations[self.nb_couches-1]
        Z = np.dot(A_prev, self.poids)





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
