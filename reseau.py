import numpy as np

class Reseau:

    def __init__(self, tailles_couches, poids_initiaux, eta):
        
        self.tailles_couches = tailles_couches #liste avec la taille de chaque couche
        self.nb_couches = len(self.tailles_couches)
        self.poids = poids_initiaux #liste de matrices np, dont la taille est nb_couches-1
        self.z = [np.zeros(taille) for taille in self.tailles_couches]
        self.activations = [np.ones(taille+1) for taille in self.tailles_couches]
        self.eta = eta #taux d'apprentissage


    def forwardProp(self, data):
        self.activations[0][:-1] = data
        for i in range(self.nb_couches-1):
            self.z[i+1] = np.dot(self.poids[i+1], self.activations[i])
            self.activations[i+1][:-1] = self.fonctionActivation(self.z[i+1])

    def forwardProp2(self, data):
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

    def descenteGradient(self, data_entrainement):
        #entrainement du réseau sur un jeu de donnée
        (xs_entrainement, ys_entrainement) = data_entrainement

        #calcul du gradient 
        gradient = [np.zeros(w.shape) for w in self.poids]
        for x in xs_entrainement:
            for y in ys_entrainement:
                self.forwardProp(x)
                new_gradient = self.backProp(x, y)
                gradient = [gradient[n] + new_gradient[n] for n in len(self.tailles_couches)]

        #mise à jour des poids
        self.poids = [self.poids[n] - self.eta * gradient[n] for n in range(len(self.tailles_couches))]
        
        


    def backProp(self, x, y):
        #renvoie le gradient de la fonction de cout (liste de np array de meme dim que self.poids)
        gradient = [np.zeros(w.shape) for w in self.poids]
        
        delta = self.deriveeCout(self.activations[-1][:-1], y) * self.deriveeActivation(self.z[-1])
        gradient[-1] = np.dot(delta, self.activations[-2].transpose())

        for l in range(2, self.taille):
            delta = np.dot(self.poids[-l+1].transpose(), delta) * self.deriveeActivation(self.z[-l])
            gradient[-l] = np.dot(delta, self.activations[-l].transpose())
        
        return gradient


        
    def fonctionActivation(self, z):
        return 1.0/(1.0+np.exp(-z)) #sigmoide
    
    def deriveeActivation(self, z):
        return self.fonctionActivation(z)*(1-self.fonctionActivation(z))

    def fonctionCout(self, x, y):
        #erreur quadratique
        return sum([(x[i]-y[i])**2 for i in range(len(x))])*0.5

    def deriveeCout(self, x, y):
        return np.ndarray([x[i]-y[i] for i in range(len(x))])


