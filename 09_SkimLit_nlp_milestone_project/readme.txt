1. Prétraitement des données
Le prétraitement des données est une étape importante dans tout projet de NLP. Dans cette partie, nous avons téléchargé le jeu de données PubMed et avons effectué les étapes suivantes:

Nettoyage du texte
Prétraitement du texte en minuscules
Suppression de la ponctuation
Suppression des stopwords
Extraction des étiquettes d'entité nommée (Named Entity Recognition - NER)
2. Préparation des données
Dans cette partie, nous avons préparé les données pour l'entraînement et l'évaluation du modèle. Les étapes suivantes ont été effectuées:

Division des données en ensembles d'entraînement, de validation et de test
Encodage des étiquettes en valeurs numériques
Création de pipelines de traitement de données pour chaque ensemble de données
3. Création des modèles
Nous avons créé cinq modèles différents pour le projet, chacun avec une architecture de réseau de neurones différente:

Modèle 0: Baseline (utilisant un classificateur binaire simple)
Modèle 1: Token Embedding personnalisé
Modèle 2: Token Embedding pré-entraîné
Modèle 3: Char Embedding personnalisé
Modèle 4: Char + Token Embedding hybride
Modèle 5: Char + Token Embedding + Part of Speech (POS) Tagging
Pour chaque modèle, nous avons entraîné et évalué sur les ensembles d'entraînement et de validation.

4. Évaluation des modèles
Dans cette partie, nous avons évalué les performances de nos modèles sur plusieurs métriques, notamment:

Précision
Rappel
F1-score
Exactitude (accuracy)
Nous avons également comparé les performances des différents modèles et avons choisi le meilleur modèle pour l'utilisation dans notre application.

5. Utilisation du modèle
Enfin, nous avons utilisé le modèle choisi pour créer une application permettant de résumer automatiquement des articles médicaux. Nous avons également exploré la possibilité de charger un modèle pré-entraîné à partir de Google Storage.