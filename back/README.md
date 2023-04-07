# TP - Serveur d'inférence simple avec FastAPI 

Pour ce Tp nous allons voir comment déployer facilement un modèle en mode API. Pour ce faire nous utiliserons fastapi.
Afin de compléter ce tp, exécutez les instructions suivantes:

## Construire un modèle de prédiction
1 - Commencer par explorer le jeu de données "heart.csv" en utilisant un jupyter notebook.
Pour plus d'informations : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

2 - Séparez votre jeu de données

3 - Construisez un premier modèle qui se base sur une seule feature et calculez ces performances avec une métrique qui vous paraît appropriée.

4 - Intégrez ce modèle dans un pipeline sklearn

5 - Exportez votre modèle à l'aide du format de votre choix (joblib, pickle, etc.)

6 - A completer :
- Ranger votre modèle dans un dossier spécialisé
- Modifier la variable DEFAULT_MODEL_PATH dans .env
- Completez la classe permettant d'intéragir avec le modèle dans app/services
- Mettre à jour le event startup dans app/core/event_handlers.py
- dans app/api/routes, modifier les routes afin de compléter le point suivant.

7 - Complétez le code main pour :
    - Créer une route "/" qui renvoie la description du jeu de données (allez sur kaggle ou renvoyez des statistiques sur le jeu de données)
    - Créer une route "/predict/" qui prend en argument (dans l'url) une valeur pour la feature de votre modèle et qui renvoie la prédiction ainsi que le score de confiance du modèle
    
    
### API

- Pour lancer l'application en local, 2 étapes sont nécessaires :
    1. installation de l'environnement python. Pour cela il faut se placer à la racine du projet et
    installer tous les requirements via la commande suivante : pip install -r requirements.txt.
    2. lancement de l'application : uvicorn app.main:app --reload --port 8443
    La documentation openapi est générée automatiquement et est accessible à l'adresse suivante : localhost:8443/docs
    
# Pour aller plus loin 
- Amélioration des performances:
  1. Reprenez votre notebook (ou prenez en un nouveau), et construisez un modèle au choix. Faites en sortes d'améliorer les performances de votre modèle en intégrant plus de features et en faisant du grid search ou random search. 
  2. Sauvegardez par la suite ce modèle (comme lors de l'étape 5). 
  3. Construisez une nouvelle route pour servir ce nouveau modèle. Vous pouvez l'appeler "/predict_{YOUR_NEW_MODEL_TYPE}". 
  4. Testez votre modèle


- Sécurisez votre API: ajouter la gestion d'une API key

### Docker 
- Conteneurisez votre backend et interragissez avec l'application depuis le backend
- conteneurisez le frontend et faite interagir les deux services à la main 
- utilisez un docker-compose pour automatiser le lancement des deux applications 
  - utilisez un volume pour gérer votre code côté backend