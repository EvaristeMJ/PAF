# SECTION de definition et d'imports a lancer au debut

import numpy as np
from playsound import playsound
import librosa

import matplotlib.pyplot as plt
import scipy 
import scipy.io



# quelques simplifications de fonctions usuelles
exp=np.exp
cos=np.cos
sin=np.sin
log=np.log

fft=np.fft.fft
ifft=np.fft.ifft
real=np.real

plot=plt.plot
stem=plt.stem
show=plt.show # force l'affichage du graphique courant
i=complex(0,1)
pi=np.pi 


audio_data,sample_rate = librosa.load('happy.mp3')
clap_data,clap_rate = librosa.load('clap.mp3')
long_clap = len(clap_data)
#audio_data = real(np.asarray((4000*[0]+[exp(2*i*pi*0.02*k) for k in range(4000)])*30))
Fe=sample_rate
Te=1/Fe

def une_colonne_spectrogramme(u,M,N,n):
    """ 
    Renvoie une colonne de spectrogramme c'est a dire la TFD de taille M
    d'une morceau de u debutant en n et de taille N multiplie par une fenetre
     de hamming """
    uu=u.copy().reshape(-1)  # on recopie sinon on risque d'ecraser 
    # construction de la fnetre
    idx=np.arange(0,N)
    w=0.54-0.46*cos(2*pi*idx/(N-1))
    # les index tels que u(m)w(n-m) non nul
    m=np.arange(n,n+N).astype(int)#COMPLETER (jusqu'au ou?)
    morceauu=uu[m]  #mrceau utile de u
    fenu=morceauu * w  #COMPLETER (quelle op�ration?)
    Uc=fft(fenu,M) # on calcule la TFD 
    Uc=abs(Uc) # on s'interesse seulement au module
    Uc=Uc[0:M//2+1] # pour un signal reel il suffit de garder la moitie
    return Uc 

def affiche_spectrogramme(u,N,M=None,nb=None,Fe=8192):
    """Affiche le specrogramme du signal u
     La taille des fenetres est N
     Si M n'est pas fourni il est pris egal � N
     nb est le pas entre deux fenetres dont on calcule la TFD 
     si nb n'est pas fourni, nb est pris egal a N/2"""
    
    if M is None:
        M=N
    if nb is None:
        nb=N/2
    # On cmmence par creer un tableau de la bonne taille don les colonnes seront
    # calculees par une_colonne_spectrogramme
    uu=u.copy().reshape(-1)
    L=len(u)
    nombre_fen=int((L-N)//nb+1)
    spectro=np.zeros((M//2+1,nombre_fen))
    for k in range(nombre_fen):
        spectro[:,k]=une_colonne_spectrogramme(u,M,N,k*nb)
    temps_debut=0
    temps_fin=nb*N*Te #COMPLETER
    freq_debut=0
    freq_fin=Fe/2 # COMPLETER

    return spectro


N=512
M=512
nb=512
spec=affiche_spectrogramme(audio_data,N,M=M,nb=nb,Fe=Fe) 

def lognozero(x):
    return log(x+10**(-10))

plt.imshow(lognozero(spec))
show()

# créer le spectrogramme des variations

(i,j)=np.shape(spec)

specdiff=np.zeros((i,j))
for k in range(1,j):    
    for l in range(i):
        specdiff[l][k] = max(0,lognozero(spec[l][k])-lognozero(spec[l][k-1]))


plt.imshow(specdiff)
show()

#Faire la moyenne sur cha<ue colonne du spectrogramme

def moy(tab):
    S=0
    for k in range(len(tab)):
        S=S+tab[k]
    S=S/len(tab)
    return(S)

X=np.arange(j)*len(audio_data)*Te/j
Y=[]
for k in range(j):
    Y.append(moy(specdiff[:, k]))
Y=np.asarray(Y)

#enlever la moyenne locale à chaque point

def withoutmoy(curb,largmoy):
    newcurb=curb
    for k in range(0,len(curb)):
        Smoy = 0
        taillecomp = 0
        for xcomp in range(max(0,k-largmoy),min(k+largmoy+1,len(curb))):
            Smoy = Smoy + curb[xcomp]
            taillecomp = taillecomp + 1
        Smoy = Smoy/taillecomp
        newcurb[k] = max(0,newcurb[k] - Smoy)      
    return(newcurb)

Y = withoutmoy(Y,10) 

plot(X,Y)
plt.xlabel('temps (s)')
show()

#trouver les maximums d'une courbe

def maxima(curb,larg):
    Max=[]
    for k in range(larg,len(curb)-larg):
        bool=True
        for xcomp in range(-larg,larg+1):
            if (curb[k+xcomp]>curb[k]):
                bool=False
        if bool:
            Max.append(k)
    print(len(Max))
    return(Max)


S=0
tps_verif = int(0.2*Fe*len(Y)/len(audio_data)) # 0.2 seconde de tolérance pour détecter les maxima
print(tps_verif)
TabMax = maxima(Y,tps_verif)

print(TabMax)

def mycorrelate(a,b):
    n=len(a)
    ac=np.asarray(n*[0])
    for k in range (n):
        S=0
        for l in range(min(n,len(b)-k)):
            S=S+a[l]*b[l+k]
        ac[k]=S
    return(ac)

Z=mycorrelate(Y,Y)
plt.plot(np.arange(len(Z)),Z)
plt.show()

def tempo_from_correl(cor):
    M = maxima(cor,tps_verif)
    print(M)
    S = 0
    for k in range(4): #on moyenne les 4 premiers recouvrements dans l'autocorrélation
        S = S + M[k+1]-M[k]
    S = S/4
    return(S*len(audio_data)*Te/len(cor))  # Avant : M[1]*len(audio_data)*Te/len(cor) ou S*len(audio_data)*Te/len(cor) 

#Ajout des beats

"""
for k in TabMax:
    ind = int(k*len(audio_data)/len(Y))
    print(ind)        
    audio_data[ind:ind+long_clap] = audio_data[ind:ind+long_clap]+10*clap_data
"""

#ajout des beats en se servant du tempo calculé
tempo = tempo_from_correl(Z)
print(tempo)
pas_ind_tempo = int(tempo*Fe)
"""
i = 0 # indice pour parcourir le tableau TabMax
k = TabMax[1]
ind = int(k*len(audio_data)/len(Y))
while ind<len(audio_data):
    if ind+long_clap > len(audio_data): #pour éviter d'arriver au bord
            break
    else:
        audio_data[ind:ind+long_clap] = audio_data[ind:ind+long_clap]+10*clap_data # on a un indice particulier pour le clap si jamais il est ajouté en fin de musique
        ind = ind + pas_ind_tempo
"""
       


i = 0 # indice pour parcourir le tableau TabMax
k = TabMax[0]
ind = int(k*len(audio_data)/len(Y))
while ind<len(audio_data):
    for l in range(4):
        if ind+long_clap > len(audio_data): #pour éviter d'arriver au bord
            break
        else:
            audio_data[ind:ind+long_clap] = audio_data[ind:ind+long_clap]+20*clap_data # on a un indice particulier pour le clap si jamais il est ajouté en fin de musique
            ind = ind + pas_ind_tempo
    k = int(ind*len(Y)/len(audio_data))
    
    while int(k*len(audio_data)/len(Y))<ind:
        if i == len(TabMax)-1:
            ind = len(audio_data)+1
            k= ind + 1
            break
        else:
            i=i+1
            k = TabMax[i]
    ind = int(k*len(audio_data)/len(Y))




#opération pour écrire la musique
normalized_data = np.interp(audio_data, (-1, 1), (-1, 1)) #normaliser les valeurs du signal à [-1,1]

audio_data_int = (normalized_data * 32767).astype(np.int16)

scipy.io.wavfile.write('output.wav', sample_rate, audio_data_int)

"""
Fe = 20
N = 200
M = 200
nb = 10
spectps = affiche_spectrogramme(Y,N,M=M,nb=nb,Fe=Fe)
plt.imshow(spectps)
show()

(itps,jtps) = np.shape(spectps)
print(itps,jtps)
for l in range(jtps):
    argmax = 0
    for k in range(itps):
        if spectps[k,l]>spectps[argmax,l]:
            argmax = k 
    print(1/(argmax*(Fe/2)/itps))
"""