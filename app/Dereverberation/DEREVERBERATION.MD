Parametres conseillés :

Taille de la fenêtre TFCT : 1024 ---
Overlap : 50% ---
Epsilon : Entre 10^-8 et 10^-10
D tel que d_t contienne les informations présentes dans les 30 à 50 premières millisecondes de réverbération

Bibliograhie :

La méthode classique la plus utilisée: 
Weighted Prediction Error (WPE) https://ieeexplore.ieee.org/abstract/document/5547558
Implémentation WPE sur python : https://github.com/fgnt/nara_wpe/blob/master/nara_wpe
Le deep évidemment: https://ieeexplore.ieee.org/document/8553141 

Simulation de datasets: 
Réponses impulsionelles simulées : bibliothèque Pyroomacoustics
Réponse mesurées dans de vraies salles https://github.com/RoyJames/room-impulse-responses 
