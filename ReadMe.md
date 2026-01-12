# TP Ollist

Le projet est composé de 2 couches :
## Couche traitement de données
Situé dans le dossier src/, composé du fichier main quiset à l'exécution du traitement des données.
Les traitmeents de données sont répartis dans les fichiers suiavnts :
### Extract
### Explore
### Transform
### Clustering

## Installation :
Après avoir cloner le dépôt :
### activation de l'environnement :
<!-- A la racine du projet, exécuter la commande suivante : -->
    source ./venv/bin/activate
### instalaltion des packages
Les packages sont renseignés dans le dossier suivant : requirements.txt
<!-- Pour installer les packages, excuter la ligne de commande suivante : -->
    pip install -r requirements.txt
## Exécution du projet :
### 1) Exécuter le point d'entrée main
    python src/main.py
### 2) Exécution du serveur Streamlit
    streamlit run dashboard/app.py