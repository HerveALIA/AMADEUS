#!/usr/bin/env python
# coding: utf-8

# # PROJET DE MACHINE LEARNING : AMADEUS

# L'objectif de ce projet est de mettre en place un modèle de Machine Learning permettant de prédire un Burn_out

# ### A- IMPORTATION DES PACKAGES ET MODULES 

# In[1]:


# Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


# ### B- IMPORTATION DES DONNEES 

# In[2]:


dataset = pd.read_csv("AMADEUS.csv")
dataset.head()


# ###  C- DATA ENGINIEERING

# In[3]:


# Affichage des colonnes ou des variables de mon dataset

print(dataset.columns.tolist())


# ##### Commentaire : 
# Mon dataset compte 10325 individus et 237 variables.
# Mais nous n'utiliserons pas toutes les 237 variables donc nous allons subseter les variables qui nous intèrèsse

# #### 1- CHOIX DE MES VARIABLES 

# In[4]:


df = dataset[['2. Quel âge avez-vous ?','SEX_M','IMC','PUBLIC_VS_PRIVE_ET_EPBNL','ANCIENNETE','11conjoint_domicile','ENFANT_BIN',
'13reconfort_entourage','14proche_aidant','service_COVID19','Anatomopathologie','Anesthésie','Biologie médicale',
'Chirurgie tête et cou','Chirurgie générale', 'Chirurgie infantile/pédiatrique', 'Chirurgie Gynéco-Obstétrique', 
'Chirurgie Maxillo-faciale','Chirurgie: Neurochirurgie','Chirurgie: Ophtalmologie', 'Chirurgie : Orthopédie', 'Chirurgie plastique', 
'Chirurgie thoracique et cardiologique','Chirurgie Viscérale et Digestive','Chirurgie Urologique', 'Chirurgie Vasculaire',
'Direction des Soins', 'Enseignement/Formation', 'Enseignement/Recherche', 'Médecine : Allergologie', 'Médecine : Cardiologie', 
'Médecine : Dermatologie', 'Médecine : Endocrinologie', 'Médecine : Hématologie', 'Médecine : Gastroentérologie', 'Médecine Générale', 
'Médecine : Génétique', 'Médecine : Gériatrie', 'Médecine : Gynécologie médicale','Médecine : Infectiologie', 'Médecine Interne',
'Médecine: Néphrologie', 'Médecine : Neurologie','Médecine Nucléaire', 'Médecine : Ophtalmologie', 'Médecine : Oncologie', 
'Médecine : Oto-Rhino-Laryngologie ( ORL)', 'Médecine : Pédiatrie (Néonatologie)', 'Médecine : Pédiatrie (hors néonatologie)', 
'Médecine Physique et de Réadaptation', 'Médecine : Pneumologie', 'Médecine: Radiologie', 'Médecine : Rhumatologie', 'Médecine : Soins palliatifs',
'Médecine : Stomatologie', 'Médecine du travail', 'Médecine: Urgences', 'Médecine: Urgences préhospitalières (SAMU)', 'Médecine Vasculaire / angiologie', 
'Psychiatrie adulte','Psychiatrie: pédopsychiatrie et enfance inadaptée', 'Soins critiques : Réanimation', 'Soins critiques : soins continus ou soins intensifs polyvalents', 'Soins critiques : soins intensifs', 
'Santé Publique', "Service d'Information Médicale", 'Service qualité et gestion des risques', '16TEMPS_COMPLET', '17POSTE_DE_NUIT', 
'18GARDES_NUIT', '19HORAIRES_CONSTANTS', '21PLANNING_2SEM_CONNU', '22DEPASSEMENT_HORAIRE_PREVU_FQC', '23WKEND_W_MOIS_NB', '24ARRET_NB_J_LASTYEAR', 
'25MALADIE_CHRONIQUE', '26ALD','KARASEK_QUANTITE_RAPIDITE', 'KARASEK_COMPLEXITE_INTENSITE','KARASEK_MORCELLEMENT_IMPREVISIBILITE',
'KARASEK_LATITUDE_MARGEMANOEUVRE', 'KARASEK_UTILISATION_COMPETENCE', 'KARASEK_DVLPT_COMPETENCE', 
'KARASEK_SOUTIEN_PRO_SUPERIEURS', 'KARASEK_SOUTIEN_PRO_COLLEGUES', 'KARASEK_SOUTIEN_EMO_SUPERIEURS', 'KARASEK_SOUTIEN_EM0_COLLEGUES', 
 '75CRAINTE_ERREUR_FQC', '76HARCELEMENT_MORAL','78HARCELEMENT_SEXUEL','80SOBD','81EDM_LIFETIME_NB','82ATD', '83ANXIO', '84PSYCHOSTIM',
 '85SUIVI_PSY4', '86SUIVI_PSYCHO', '87NB_CIG', '88CAFE_TASSES','DETA_CUTOFF2','METS_MIN_SEMAINE',
'120. Au cours des 30 derniers jours, au bout de combien de temps ( en minutes) vous êtes-vous généralement endormi(e) le soir?', 
'122.DUREE_SOMM_CONTINU','PSQI_TBSOMMEIL_CONTINU',  'PSQI_QUALITE_CONTINU', 
'125. Au cours des 30 derniers jours, combien de fois avez-vous pris des médicaments pour mieux dormir (médicaments prescrits par votre médecin ou vendus sans ordonnance) ?',
'126. Au cours des 30 derniers jours, combien de fois avez-vous eu des difficultés à rester éveillé(e) en conduisant, en mangeant, ou en participant à des activités avec d’autres personnes ?',
'127. Au cours des 30 derniers jours, combien vous a-t-il été difficile d’être suffisamment motivé(e) pour mener à bien vos activités ?','BURNOUT_BIN']]


# In[5]:


# Récupération des noms des colonnes contenant le mot "KARASEK"
karasek_cols = [col for col in dataset.columns if "KARASEK" in col]

# Affichage des noms des colonnes contenant le mot "KARASEK"
print(karasek_cols)
#
print(len(karasek_cols))


# In[6]:


df.shape


# ### df Update

# In[7]:


df = dataset[['INTERNE', 'IDE', 'AS','2. Quel âge avez-vous ?','SEX_M','NON_BINAIRE','SURPOIDS_OBESITE','PUBLICETEPBNL_VS_PRIVE','ANCIENNETE','11conjoint_domicile','ENFANT_BIN',
'13reconfort_entourage','14proche_aidant','service_COVID19','Anatomopathologie','Anesthésie','CADRE_CADRESUP_ALL', 'PHARMACIEN', 'PSYCHO', 'SAGEFEMME', 'KINE', 'ERGO', 'DIRECTEURSOINS',
'SANITAIRE_VS_MEDICOSOCIAL',  'Psychiatrie adulte','Psychiatrie: pédopsychiatrie et enfance inadaptée',  
'Santé Publique', "Service d'Information Médicale", '16TEMPS_COMPLET', '17POSTE_DE_NUIT', '18GARDES_NUIT', '19HORAIRES_CONSTANTS', '21PLANNING_2SEM_CONNU', '22DEPASSEMENT_HORAIRE_PREVU_FQC', 
'23WKEND_W_MOIS_NB', '24ARRET_NB_J_LASTYEAR', '25MALADIE_CHRONIQUE','KARASEK_AXE1_DEMANDE_PSYCHOLOGIQUE','KARASEK_AXE3_SOUTIEN_SOCIAL','KARASEK_AXE2_LATITUDE_DECISIONELLE','75CRAINTE_ERREUR_FQC', '76HARCELEMENT_MORAL','78HARCELEMENT_SEXUEL','80SOBD','81EDM_LIFETIME_NB','82ATD', '83ANXIO',
'85SUIVI_PSY4', '86SUIVI_PSYCHO', '87NB_CIG', '88CAFE_TASSES','DETA_CUTOFF2','METS_MIN_SEMAINE',
'120. Au cours des 30 derniers jours, au bout de combien de temps ( en minutes) vous êtes-vous généralement endormi(e) le soir?', 
'122.DUREE_SOMM_CONTINU','PSQI_TBSOMMEIL_CONTINU',  'PSQI_QUALITE_CONTINU', 
'126. Au cours des 30 derniers jours, combien de fois avez-vous eu des difficultés à rester éveillé(e) en conduisant, en mangeant, ou en participant à des activités avec d’autres personnes ?',
'127. Au cours des 30 derniers jours, combien vous a-t-il été difficile d’être suffisamment motivé(e) pour mener à bien vos activités ?','BURNOUT_BIN',
'SERVICE_CHIR', 'SERVICE_SPEMED', 'SOINS_CRITIQUES','HYPNOTIQUES']]


# In[8]:


sorted_columns = sorted(df.columns)
print(sorted_columns)


# In[9]:


df.shape


# ## Vérification de mon df update

# In[10]:


# Créez une liste pour stocker les noms de colonnes en majuscules
uppercase_columns = []

# Parcourez toutes les colonnes du dataset
for column in df.columns:
    # Vérifiez si le nom de la colonne est en majuscules
    if column.isupper():
        # Ajoutez la colonne à la liste des colonnes en majuscules
        uppercase_columns.append(column)

# Affichez les colonnes en majuscules
print("Colonnes en majuscules :", uppercase_columns)

# Créez un sous-ensemble de données contenant uniquement les colonnes en majuscules
uppercase_data = df[uppercase_columns]

# Affichez le sous-ensemble de données sous forme de tableau
for i, column in enumerate(uppercase_columns, start=1):
    print(f"{i}. {column}")


# # % de chaque N 

# In[11]:


# boucler à travers chaque colonne
for column in df.columns:
    # vérifier si la colonne est binaire
    if df[column].nunique() == 2 and column != 'BURNOUT_BIN':
        
       
# compter le nombre d'occurrences de chaque valeur dans la colonne
        counts = df[column].value_counts()
        # calculer le pourcentage de chaque valeur
        percent_Y0 = counts[0] / len(df) * 100
        percent_Y1 = counts[1] / len(df) * 100
        # imprimer les résultats
        print(f"Variable {column}:")
        print(f"Pourcentage de BURNOUT_BIN=0 : {percent_Y0 :.2f}%")
        print(f"Pourcentage de BURNOUT_BIN=1 : {percent_Y1:.2f}%")


# ## Pour les variables non binaires 

# In[12]:


# lister toutes les colonnes du dataframe
columns = df.columns.tolist()


columns
# boucler à travers chaque colonne
for column in columns:
    # vérifier si la colonne est non binaire
    if df[column].nunique() > 2 and column != 'BURNOUT_BIN':
        # grouper le dataframe par la colonne et compter les occurrences de chaque catégorie pour chaque valeur de Y
        counts = df.groupby([column, 'BURNOUT_BIN']).size().unstack(fill_value=0)
        # diviser par le nombre total d'observations pour obtenir le pourcentage
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        # calculer la moyenne des pourcentages de chaque catégorie en fonction des valeurs de la variable de réponse
        means = percentages.mean(axis=0)
        # imprimer les résultats
        print(f"Variable {column}:")
        print(means)


# In[13]:


# Trouver les colonnes en double
duplicated_cols = df.columns[df.columns.duplicated()]

# Afficher les colonnes en double
print("Colonnes en double :")
for col in duplicated_cols:
    print(col)


# #### Vérification de la variable nombre d'enfant en continue

# In[14]:


# Vérifier le type de données d'une variable
type_de_variable = df['ENFANT_BIN'].dtype

# Vérifier si la variable est continue ou non
if type_de_variable == 'float64' or type_de_variable == 'int64':
    print('La variable est continue')
else:
    print('La variable n\'est pas continue')


# In[15]:


df['ENFANT_BIN']


# ### Verification : qu'il n'y ait pas dans le modèle final des variables binaires et continues décrivant la même chose 

# In[16]:


# Sélectionner les variables binaires
binary_vars = df.loc[:, (df.nunique() <= 2)]

# Sélectionner les variables continues
continuous_vars = df.select_dtypes(include=['float64', 'int64'])

# Afficher les statistiques descriptives pour les variables continues
print("Statistiques descriptives des variables continues :")
print(continuous_vars.describe())

# Afficher le nombre de valeurs uniques pour chaque variable binaire
print("\nNombre de valeurs uniques pour chaque variable binaire :")
for col in binary_vars:
    print(col, binary_vars[col].value_counts())


# In[17]:


# Récupérer les noms de colonnes
cols = df.columns

# Initialiser une liste vide pour stocker les noms de colonnes qui sont en double
duplicates = []

# Boucle à travers les noms de colonnes
for i, col in enumerate(cols):

    # Vérifier si la colonne est booléenne
    if df[col].dtype == 'bool':

        # Vérifier si la colonne est aussi continue
        if df[col].nunique() == 2 and df[col].dropna().isin([0, 1]).all():
            duplicates.append(col)

# Afficher les noms de colonnes en double
if duplicates:
    print(f"Les variables suivantes sont binaires et continues et décrivent la même chose : {duplicates}")
else:
    print("Aucune variable binaire et continue ne décrit la même chose.")


# In[18]:


df


# In[19]:


df.shape


# Après le choix de nos variables nous voyons que nous avons fait 129 variables mais toujours avec 10325 individus

# #### 2-  Renommer les variables avec un tableau de correspondance

# In[20]:


df1 = df.copy()  # Create copy of DataFrame


# ##### Commentaire : 
# Nous avons vérifié le type de nos variables afin de nous assurer qu'ils sont tous des numériques ou des floatés mais ci-dessus nous voyons qu'il y a 4 variables qui sont des objets donc nous allons changer leur type

# In[21]:


df1.select_dtypes(object).columns


# ### 2éme variable :  ANCIENNETE

# In[22]:


df1['ANCIENNETE']=df1['ANCIENNETE'].apply(lambda x: str(x).replace(",", "."))
df1['ANCIENNETE']=df1['ANCIENNETE'].astype(float)


# In[23]:


df1['ANCIENNETE']


# ### 3éme variable :  METS_MIN_SEMAINE

# In[24]:


df1['METS_MIN_SEMAINE']


# In[25]:


df1['METS_MIN_SEMAINE']=df1['METS_MIN_SEMAINE'].apply(lambda x: str(x).replace(",", "."))
df1['METS_MIN_SEMAINE']=df1['METS_MIN_SEMAINE'].astype(float)


# In[26]:


df1['METS_MIN_SEMAINE']


# ### 4éme variable :  122.DUREE_SOMM_CONTINU

# In[27]:


df1['122.DUREE_SOMM_CONTINU']


# In[28]:


df1['122.DUREE_SOMM_CONTINU']=df1['122.DUREE_SOMM_CONTINU'].apply(lambda x: str(x).replace(",", "."))
df1['122.DUREE_SOMM_CONTINU']=df1['122.DUREE_SOMM_CONTINU'].astype(float)


# In[29]:


df1['122.DUREE_SOMM_CONTINU']


# ##### Commentaire : 
# Donc nous avons toutes nos variables qui sont des variables quantitatives 

# #### 3-  Gestion des données manquantes

# In[30]:


df1.isnull().sum()[df1.isnull().sum()>0]


# ##### Commentaire : 
# Nous avons 3 variables dans notre dataframe qui ont respectivement 118, 4 et 140 données manquantes

# In[31]:


def missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Pourcentage'])
    #Affiche que les variables avec des na
    #display (missing_data[(percent>0)])
    return missing_data[(percent>0)]


# In[32]:


missing_values(df1).T
#le T c'est pour transposer la table 


# ### Machine Learning Superviséé

# ### 1ère Partie  : Data_préprocessing

# #### Imputation par la moyenne

# In[33]:


# Etant donné que j'ai des données manquantes dans mon dataframe je les remplace par la moyenne de la variable concernée 


# In[34]:


df1['122.DUREE_SOMM_CONTINU'].mean()


# In[35]:


df1.fillna(df1.mean(),inplace=True)


# In[36]:


missing_values(df1).T
#le T c'est pour transposer la table 


# In[37]:


df1.select_dtypes(object).columns


# In[38]:


# 1) Créer une matrice des variables indépendantes et le vecteur de la variable dépendante.
# X est la matrice et Y est le vecteur
# PS: Astuce : Très souvent la dernière colonne est la colonne dépendante le y à prédire
# La matrice des variables indépendantes est aussi appeelée matrice de features


# In[39]:


df1


# In[40]:


df1.shape


# In[41]:


print(df1.columns.tolist())


# # Gestion des variables binaires et non binaires 

# In[42]:


binary_vars = []
non_binary_vars = []
for col in df1.columns:
    if df1[col].nunique() == 2:
        binary_vars.append(col)
    else:
        non_binary_vars.append(col)


# In[43]:


# Afficher les colonnes binaires
print('Colonnes binaires :', binary_vars)


# In[44]:


print(len(non_binary_vars))
print(len(binary_vars))


# In[45]:


# Afficher les colonnes non_binaires
print('Colonnes non binaires :', non_binary_vars)


# In[46]:


from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()


# In[47]:


# Standardisation des variables non binaires
scaler1 = StandardScaler()
df1[non_binary_vars] = scaler1.fit_transform(df1[non_binary_vars])


# In[48]:


# Concaténation des variables binaires et non binaires
df2 = pd.concat([df1[binary_vars], df1[non_binary_vars]], axis=1)


# In[49]:


df2


# In[50]:


df2.shape


# In[51]:


print(df2[non_binary_vars].describe())


# In[52]:


df2['BURNOUT_BIN']


# In[53]:


# 1) Créer une matrice des variables indépendantes et le vecteur de la variable dépendante.
# X est la matrice et Y est le vecteur
# PS: Astuce : Très souvent la dernière colonne est la colonne dépendante le y à prédire
# La matrice des variables indépendantes est aussi appeelée matrice de featuresµ

X = df2.drop('BURNOUT_BIN', axis=1)  # Supprimer la colonne "target" de la matrice X
Y = df2['BURNOUT_BIN']              # Sélectionner uniquement la colonne "target" pour Y

# Diviser les données en 


# In[54]:


X


# In[55]:


Y


# ### e) Séparation du dataset en training_set et en test_set

# In[56]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state= 0)


# In[57]:


#Vérifionsla proportion des 1 et des 0 dans mon Y


# In[58]:


np.unique(Y,return_counts=True)


# In[59]:


print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


# In[60]:


pd.DataFrame(X_train).head()


# # TEST DES MODELES

# ###                           ML SUPERVISE

# ## MODELES CHOISIS : LOGREG,SVM,RF,GB,ANN

# # LOGREG

# La régression logistique est une méthode de classification supervisée en Machine Learning. Elle est utilisée pour prédire une variable cible binaire (par exemple oui/non, vrai/faux, 0/1) en fonction de plusieurs variables explicatives.
# 
# Le modèle de régression logistique utilise une fonction logistique pour modéliser la relation entre les variables explicatives et la variable cible. Cette fonction de régression produit une probabilité entre 0 et 1 qui est utilisée pour classer chaque observation dans une des deux catégories possibles.

# ### RFE

# In[61]:


# Pour cela il faut importer une classe qui s'appelle linear.regression

from sklearn.linear_model import LogisticRegression

classifierlogreg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear',random_state = 0)

#Underfitting : MauvaiseApprentissage
#Overfitting :  SurApprentissage


# In[62]:


from sklearn.feature_selection import RFE

# Créer un objet RFE
rfe = RFE(estimator=classifierlogreg, n_features_to_select=13)

# Adapter l'objet RFE aux données
rfe.fit(X_train, Y_train )

# Récupère le masque des entités sélectionnées
feature_mask = rfe.get_support()

# Récupère le noms des entités sélectionnées
selected_feature_names = X_train.columns[feature_mask]

# Créer un nouveau dataframe avec les noms des entités sélectionnées
selected_features_df = pd.DataFrame(selected_feature_names, columns=['Selected Features'])

# Imprimer les noms des entités sélectionnées
print(selected_features_df)


# ### OR, CI, Pvalue des featurselections

# In[63]:


import statsmodels.api as sm
# Sélectionner les variables sélectionnées par RFE
selected_features = ['SEX_M','ENFANT_BIN','PHARMACIEN','SAGEFEMME','SANITAIRE_VS_MEDICOSOCIAL','86SUIVI_PSYCHO','DETA_CUTOFF2',
                  'KARASEK_AXE1_DEMANDE_PSYCHOLOGIQUE','KARASEK_AXE3_SOUTIEN_SOCIAL','KARASEK_AXE2_LATITUDE_DECISIONELLE','75CRAINTE_ERREUR_FQC',
                    'PSQI_QUALITE_CONTINU','127. Au cours des 30 derniers jours, combien vous a-t-il été difficile d’être suffisamment motivé(e) pour mener à bien vos activités ?']
X_train_rfe = X_train[selected_features]

# Ajouter une constante à X_train_rfe
X_train_rfe = sm.add_constant(X_train_rfe)

# Créer un objet de modèle Logit avec les données X_train_rfe et Y_train
logit_model_rfe = sm.Logit(Y_train, X_train_rfe)

# Adapter le modèle aux données
result_rfe = logit_model_rfe.fit()

# Récupérer les coefficients, les IC et les p-values
params_rfe = result_rfe.params
conf_rfe = result_rfe.conf_int()
conf_rfe['OR'] = params_rfe
conf_rfe.columns = ['Lower CI', 'Upper CI', 'OR']
conf_rfe = np.exp(conf_rfe)
conf_rfe['p-value'] = result_rfe.pvalues

# Formater les valeurs pour l'affichage
pd.options.display.float_format = '{:.2f}'.format

# Créer un tableau avec les résultats
results_table_rfe = pd.DataFrame(index=X_train_rfe.columns)
results_table_rfe.index.name = 'Variable'
results_table_rfe['OR'] = conf_rfe['OR']
results_table_rfe['Lower CI'] = conf_rfe['Lower CI']
results_table_rfe['Upper CI'] = conf_rfe['Upper CI']
results_table_rfe['p-value'] = conf_rfe['p-value']

# Afficher le tableau de résultats
print(results_table_rfe.to_string())


# ### LES PARAMETRES

# 
# 
# #penalty : Le type de régularisation à utiliser. Les options courantes sont l1, l2, elasticnet et none. Par défaut, la régression logistique utilise une régularisation L2 (également appelée régularisation Ridge) en spécifiant penalty='l2'
# 
# #C : L'inverse de la force de régularisation. Une valeur plus élevée de C signifie une régularisation plus faible, ce qui permet au modèle de s'adapter davantage aux données d'entraînement. Par défaut, C=1.0
# 
# #solver : L'algorithme utilisé pour optimiser les coefficients du modèle. Les options courantes sont newton-cg, lbfgs, liblinear, sag et saga.
# 
# #Le choix de l'algorithme dépend de la taille du jeu de données et des caractéristiques spécifiques du problème de classification. Par défaut, solver='lbfgs'.
# 

# In[64]:


# compter le nombre d'occurrences de chaque valeur dans la colonne Y
counts = df['SEX_M'].value_counts()

# calculer le pourcentage de chaque valeur
percentage_Y0 = counts[0] / len(df) * 100
percentage_Y1 = counts[1] / len(df) * 100

print(f"Pourcentage de Y=0: {percentage_Y0:.2f}%")
print(f"Pourcentage de Y=1: {percentage_Y1:.2f}%")


# In[65]:


# compter le nombre d'occurrences de chaque valeur dans la colonne Y
counts = df['PHARMACIEN'].value_counts()

# calculer le pourcentage de chaque valeur
percentage_Y0 = counts[0] / len(df) * 100
percentage_Y1 = counts[1] / len(df) * 100

print(f"Pourcentage de Y=0: {percentage_Y0:.2f}%")
print(f"Pourcentage de Y=1: {percentage_Y1:.2f}%")


# In[66]:


# compter le nombre d'occurrences de chaque valeur dans la colonne Y
counts = df['SAGEFEMME'].value_counts()

# calculer le pourcentage de chaque valeur
percentage_Y0 = counts[0] / len(df) * 100
percentage_Y1 = counts[1] / len(df) * 100

print(f"Pourcentage de Y=0: {percentage_Y0:.2f}%")
print(f"Pourcentage de Y=1: {percentage_Y1:.2f}%")


# In[67]:


# compter le nombre d'occurrences de chaque valeur dans la colonne Y
counts = df['IDE'].value_counts()

# calculer le pourcentage de chaque valeur
percentage_Y0 = counts[0] / len(df) * 100
percentage_Y1 = counts[1] / len(df) * 100

print(f"Pourcentage de Y=0: {percentage_Y0:.2f}%")
print(f"Pourcentage de Y=1: {percentage_Y1:.2f}%")


# In[68]:


# Calculer la fréquence de la variable Y
freq_y = dataset['BURNOUT_BIN'].value_counts(normalize=True)

# Afficher les résultats
print(freq_y)

#
total_freq = df['BURNOUT_BIN'].value_counts().sum()
print(total_freq)


# ### Faire de nouvelles prédictions 

# In[69]:


Y_pred = rfe.predict(X_test)
Y_pred


# In[70]:


print(Y_test)
print(Y_pred)


# In[71]:


with np.printoptions(threshold=np.inf):
    print(Y_test)


# In[72]:


with np.printoptions(threshold=np.inf):
    print(Y_pred)


# ## Matrice de confusion

# In[73]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(Y_test, Y_pred)
CM


# In[74]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                CM.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     CM.flatten()/np.sum(CM)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(CM, annot=labels, fmt='', cmap='Blues')


# ### Courbe roc 

# La courbe ROC (Receiver Operating Characteristic) est un graphique du taux de vrais positifs par rapport au taux de faux positifs. Il montre le compromis entre sensibilité et spécificité.

# In[75]:


from sklearn import metrics
y_pred_proba = rfe.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="df2, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# ### ACCURACY RATE, ERROR RATE & F1-SCORE

# In[76]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics


# In[77]:


Accuracy_Rate = accuracy_score(Y_test, Y_pred)
Error_rate = 1 - Accuracy_Rate
F1_score_logreg = f1_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
CK = cohen_kappa_score (Y_test,Y_pred)
MC = matthews_corrcoef(Y_test,Y_pred)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)

print("precision : {:.2f}".format(precision))
print("recall : {:.2f}".format(recall))
print("Accuracy rate: ", Accuracy_Rate)
print("Error rate: ", Error_rate)
print("F1_score: ", F1_score_logreg)
print("CK:", CK)
print("MC:", MC)
print("AUC:", auc)


# In[78]:


# create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "Error Rate", "F1 Score", "CK", "MC","AUC"]
metric_values = [precision, recall, Accuracy_Rate, Error_rate, F1_score_logreg, CK, MC,auc]

# create a bar chart
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(metric_names, metric_values)
ax.set_ylabel('Value')
ax.set_ylim([0,1])
ax.set_title('Performance Metrics')
plt.show()


# # Construction du modèle SVM (Support Vector Machine) : modèle linéaire

# Le modèle SVM (Support Vector Machine) est un modèle d'apprentissage supervisé utilisé pour la classification et la régression. L'objectif de ce modèle est de trouver une frontière de décision qui sépare les données en deux classes distinctes avec la plus grande marge possible.
# 
# Pour ce faire, le modèle SVM utilise un hyperplan qui sépare les données en deux classes en maximisant la marge, c'est-à-dire la distance entre l'hyperplan et les points les plus proches de chaque classe. Les points les plus proches de l'hyperplan sont appelés vecteurs de support, d'où le nom de Support Vector Machine.
# 
# Le modèle SVM peut être utilisé pour la classification binaire et multiclasse, ainsi que pour la régression. Pour la classification binaire, le modèle SVM linéaire utilise une fonction d'activation de type "step function" tandis que les modèles SVM non linéaires utilisent des fonctions d'activation non linéaires telles que le noyau gaussien.
# 
# Le modèle SVM a de bonnes performances dans la classification de données linéairement et non linéairement séparables. Cependant, il peut être sensible aux données bruyantes et peut être coûteux en termes de temps de calcul pour de grandes quantités de données.

# In[79]:


from sklearn.svm import SVC

classifier2 = SVC(kernel= 'linear', C=0.01, gamma='scale',random_state = 0, probability= True)


# In[80]:


from sklearn.feature_selection import RFE

# Créer un objet RFE
rfe2 = RFE(estimator=classifier2, n_features_to_select=13)

# Adapter l'objet RFE aux données
rfe2.fit(X_train, Y_train )


# In[81]:


# Get the mask of selected features
feature_mask2 = rfe2.get_support()

# Get the names of the selected features
selected_feature_names2 = X_train.columns[feature_mask]

# Create a new dataframe with the selected feature names
selected_features_df2 = pd.DataFrame(selected_feature_names2, columns=['Selected Features'])

# Print the selected feature names
print(selected_features_df2)


# ### LES PARAMETRES

# #Noyau (kernel) : le noyau est une fonction mathématique qui transforme les données d'entrée (vecteurs) dans un espace de dimension supérieure, où elles sont plus facilement séparables. Les noyaux les plus couramment utilisés sont le noyau linéaire (Linear kernel), le noyau RBF (Radial basis function kernel), le noyau polynomial (Polynomial kernel), et le noyau sigmoidal (Sigmoid kernel).
# 
# #C : C est un paramètre de régularisation qui contrôle la pénalité appliquée aux erreurs de classification. Un C faible permettra une certaine classification incorrecte en échange d'une surface de décision plus simple, tandis qu'un C élevé donnera une surface de décision plus complexe, en essayant de classifier toutes les données correctement. C agit donc comme une mesure d'équilibre entre la simplicité de la surface de décision et la précision de classification.
# 
# #Gamma : Gamma est un paramètre qui définit la "distance" d'influence de chaque point d'entraînement dans la fonction de décision. Plus gamma est grand, plus la surface de décision sera "pointue" et plus chaque point d'entraînement aura une influence étendue. Un gamma plus faible aura l'effet inverse, conduisant à une surface de décision plus douce et plus régulière.
# 
# #Probability est un paramètre optionnel pour les modèles SVM de la librairie scikit-learn. Ce paramètre permet d'activer la sortie des probabilités des prédictions du modèle.
# 
# Par défaut, les modèles SVM de scikit-learn ne fournissent pas les probabilités des prédictions, mais plutôt une prédiction binaire (0 ou 1) pour chaque exemple. Si le paramètre probability est activé, le modèle SVM utilise une méthode de calibration pour estimer les probabilités des prédictions.

# ### Faire de nouvelles prédictions 

# In[82]:


Y2_pred = rfe2.predict(X_test)
print(Y_test)
print(Y2_pred)


# In[83]:


with np.printoptions(threshold=np.inf):
    print(Y_test)


# In[84]:


with np.printoptions(threshold=np.inf):
    print(Y2_pred)


# ### MATRICE DE CONFUSION

# In[85]:


from sklearn.metrics import confusion_matrix
CM2 = confusion_matrix(Y_test, Y2_pred)
CM2


# In[86]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                CM2.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     CM2.flatten()/np.sum(CM2)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(CM2, annot=labels, fmt='', cmap='coolwarm')


# In[87]:


print(Y_test.shape)
print(CM2)


# # Courbe roc 

# In[88]:


y_pred_proba2 = rfe2.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba2)
auc = metrics.roc_auc_score(Y_test, y_pred_proba2)
plt.plot(fpr,tpr,label="df2, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# ### ACCURACY RATE, ERROR RATE & F1-SCORE

# In[89]:


Accuracy_Rate2 = accuracy_score(Y_test, Y2_pred)
Error_rate2 = 1 - Accuracy_Rate2  # Change variable name to Error_rate2
F1_score_logreg2 = f1_score(Y_test, Y2_pred)
precision2 = precision_score(Y_test, Y2_pred)
recall2 = recall_score(Y_test, Y2_pred)
CK2 = cohen_kappa_score(Y_test, Y2_pred)
MC2 = matthews_corrcoef(Y_test, Y2_pred)
auc2 = metrics.roc_auc_score(Y_test, y_pred_proba2)

print("precision : {:.2f}".format(precision2))
print("recall : {:.2f}".format(recall2))
print("Accuracy rate: ", Accuracy_Rate2)
print("Error rate: ", Error_rate2)
print("F1_score: ", F1_score_logreg2)
print("CK:", CK2)
print("MC:", MC2)
print("AUC:", auc2)


# In[90]:


# create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "Error Rate", "F1 Score", "CK", "MC","AUC"]
metric_values = [precision2, recall2, Accuracy_Rate2, Error_rate2, F1_score_logreg2, CK2, MC2,auc2]

# create a bar chart
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(metric_names, metric_values)
ax.set_ylabel('Value')
ax.set_ylim([0,1])
ax.set_title('Performance Metrics')
plt.show()


# # Construction du modèle Random Forest : modèle non linéaire

# Le modèle de Random Forest est un algorithme d'apprentissage supervisé en Machine Learning qui utilise une combinaison d'arbres de décision pour créer un modèle prédictif.
# 
# L'idée est de construire plusieurs arbres de décision, en utilisant différents échantillons d'entraînement et des sous-ensembles de fonctionnalités, afin de créer un ensemble de modèles qui travaillent ensemble pour donner une prédiction précise. Les prédictions de chaque arbre sont ensuite agrégées pour donner une prédiction finale.
# 
# Le modèle de Random Forest peut être utilisé pour la classification et la régression, et est particulièrement utile pour les ensembles de données avec de nombreuses caractéristiques, y compris des variables continues et catégorielles. Il est également robuste aux valeurs manquantes et aux données bruyantes.
# 
# Les avantages du modèle de Random Forest sont sa précision de prédiction, sa capacité à traiter les données manquantes et bruyantes, et sa capacité à gérer de grandes quantités de données. Cependant, il peut être plus lent à entraîner que d'autres modèles et peut être plus difficile à interpréter en raison de la complexité de l'ensemble d'arbres de décision.

# In[91]:


###Boruta


# In[92]:


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# In[93]:


X_train_np = X_train.to_numpy()


# In[94]:


# Initialiser le modèle de classification Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# let's initialize Boruta
feat_selector = BorutaPy(
    verbose=2,
    estimator=rf,
    n_estimators='auto',
    max_iter=10  # number of iterations to perform
)

# train Boruta
# N.B.: X and y must be numpy arrays
#feat_selector.fit(np.array(X_train), np.array(Y_train))


# In[95]:


# Select features using BorutaPy
feat_selector.fit(X_train.values, Y_train.values.ravel())

# Get selected features
X_train_selected = feat_selector.transform(X_train.values)
X_test_selected = feat_selector.transform(X_test.values)

# Train a machine learning model using the selected features
model = RandomForestClassifier()
model.fit(X_train_selected, Y_train)

# Get selected features
X_train_selected = feat_selector.transform(X_train.values)

# Print selected feature names
print(selected_feature_names)


# ### LES PARAMETRES 

# Les paramètres utilisés dans ce modèle Random Forest sont :
# 
# #n_estimators : le nombre d'arbres de décision à construire. Plus ce nombre est grand, plus le modèle sera robuste, mais cela peut augmenter le temps d'entraînement. La valeur par défaut est 100.
# 
# #max_depth : la profondeur maximale de chaque arbre de décision. Une valeur plus grande peut permettre au modèle de mieux s'adapter aux données d'entraînement, mais peut également entraîner un surajustement. La valeur par défaut est None, ce qui signifie que les arbres sont développés jusqu'à ce que toutes les feuilles soient pures ou que le nombre minimal d'échantillons requis pour diviser une feuille soit atteint.
# 
# #min_samples_split : le nombre minimum d'échantillons requis pour diviser un nœud interne. Si le nombre d'échantillons dans un nœud est inférieur à ce paramètre, le nœud ne sera pas divisé. Une valeur plus grande peut empêcher le modèle de surajuster, mais peut également réduire les performances. La valeur par défaut est 2.
# 
# #min_samples_leaf : le nombre minimum d'échantillons requis pour être dans une feuille. Si une feuille contient moins d'échantillons que ce paramètre, la division du nœud est annulée. Comme min_samples_split, une valeur plus grande peut empêcher le modèle de surajuster, mais peut également réduire les performances. La valeur par défaut est 1.
# 
# #random_state : permet de fixer la graine aléatoire pour la reproductibilité des résultats. La valeur par défaut est None, ce qui signifie que chaque exécution peut donner des résultats différents.
# 

# ### Faire des prédictions sur l'ensemble de test

# In[96]:


Y6_pred = model.predict(X_test_selected)
print(Y_test)
print(Y6_pred)


# In[97]:


with np.printoptions(threshold=np.inf):
    print(Y_test)


# In[98]:


with np.printoptions(threshold=np.inf):
    print(Y6_pred)


# In[99]:


from sklearn.metrics import confusion_matrix
CM6 = confusion_matrix(Y_test, Y6_pred)
CM6


# In[100]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                CM6.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     CM6.flatten()/np.sum(CM6)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(CM6, annot=labels, fmt='', cmap='Pastel1')


# In[101]:


print(Y_test.shape)
print(CM6)


# # Courbe roc 

# In[102]:


y_pred_proba6 = model.predict_proba(X_test_selected)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba6)
auc = metrics.roc_auc_score(Y_test, y_pred_proba6)
plt.plot(fpr,tpr,label="df2, auc="+str(auc))
plt.legend(loc=4)  #Y6_pred = model.predict(X_test_selected)
plt.show()
# La courbe roc montre la spécifité et la sensibilité


# ### ACCURACY RATE, ERROR RATE & F1-SCORE

# In[103]:


Accuracy_Rate6 = accuracy_score(Y_test, Y6_pred)
Error_rate6 = 1 - Accuracy_Rate6  # Change variable name to Error_rate2
F1_score_logreg6 = f1_score(Y_test, Y6_pred)
precision6 = precision_score(Y_test, Y6_pred)
recall6 = recall_score(Y_test, Y6_pred)
CK6 = cohen_kappa_score(Y_test, Y6_pred)
MC6 = matthews_corrcoef(Y_test, Y6_pred)
auc3 = metrics.roc_auc_score(Y_test, y_pred_proba6)

print("precision : {:.2f}".format(precision6))
print("recall : {:.2f}".format(recall6))
print("Accuracy rate: ", Accuracy_Rate6)
print("Error rate: ", Error_rate6)
print("F1_score: ", F1_score_logreg6)
print("CK:", CK6)
print("MC:", MC6)
print("AUC:", auc3)


# In[104]:


# create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "Error Rate", "F1 Score", "CK", "MC","AUC"]
metric_values = [precision6, recall6, Accuracy_Rate6, Error_rate6, F1_score_logreg6, CK6, MC6,auc3]

# create a bar chart
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(metric_names, metric_values)
ax.set_ylabel('Value')
ax.set_ylim([0,1])
ax.set_title('Performance Metrics')
plt.show()


# # Construction du modèle Gradient Boosting : modèle non linéaire

# Le Gradient Boosting est une technique d'apprentissage automatique utilisée pour résoudre des problèmes de régression et de classification. C'est une méthode d'ensemble qui combine plusieurs modèles de prédiction plus faibles (souvent des arbres de décision) pour former un modèle plus fort.
# 
# Le Gradient Boosting fonctionne en ajustant successivement les modèles pour minimiser les erreurs de prédiction du modèle précédent. À chaque étape, le Gradient Boosting calcule les résidus (différences entre les valeurs réelles et prédites) du modèle précédent, puis ajuste un nouveau modèle pour prédire ces résidus. Le résultat final est une combinaison pondérée de tous les modèles.
# 
# Le Gradient Boosting est efficace pour résoudre les problèmes de classification et de régression non linéaires, et est largement utilisé dans de nombreux domaines tels que la finance, la biologie et l'analyse de données. Cependant, il peut être assez sensible aux hyperparamètres et nécessite une optimisation minutieuse pour atteindre les meilleures performances.

# In[105]:


from sklearn.ensemble import GradientBoostingClassifier


# In[106]:


# Initialisation du modèle de classification Gradient Boosting
gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, min_samples_leaf=1, max_features=None,
                                           max_depth=3, min_samples_split=2, random_state=42)

# Entraînement du modèle sur l'ensemble d'entraînement
#gb_classifier.fit(X_train, Y_train)


# In[107]:


# let's initialize Boruta
feat_selector2 = BorutaPy(
    verbose=2,
    estimator=gb,
    n_estimators='auto',
    max_iter=10  # number of iterations to perform
)

# train Boruta
# N.B.: X and y must be numpy arrays
#feat_selector.fit(np.array(X_train), np.array(Y_train))


# In[108]:


# Select features using BorutaPy
feat_selector2.fit(X_train.values, Y_train.values.ravel())

# Get selected features
X_train_selected = feat_selector2.transform(X_train.values)
X_test_selected = feat_selector2.transform(X_test.values)

# Train a machine learning model using the selected features
model2 = GradientBoostingClassifier()
model2.fit(X_train_selected, Y_train)

# Get selected features
X_train_selected = feat_selector2.transform(X_train.values)
# Print selected feature names
print(selected_feature_names)


# #### LES PARAMETRES 

# #n_estimators : C'est le nombre d'arbres de décision dans le modèle. Plus il y a d'arbres, plus le modèle peut apprendre de relations complexes entre les variables d'entrée et les sorties souhaitées. Cependant, cela peut également rendre le modèle plus complexe et augmenter le risque de surapprentissage. Il est donc important de trouver un équilibre entre le nombre d'estimateurs et les performances de prédiction.
# 
# #learning_rate : Il s'agit du taux d'apprentissage, qui détermine la vitesse à laquelle le modèle apprend des erreurs de prédiction. Un taux d'apprentissage plus élevé signifie que le modèle s'adapte plus rapidement aux erreurs, mais cela peut également augmenter le risque de surapprentissage. Un taux d'apprentissage plus faible signifie que le modèle prend plus de temps pour s'adapter aux erreurs, mais cela peut également aider à prévenir le surapprentissage.
# 
# #max_depth : C'est la profondeur maximale de chaque arbre de décision. Une profondeur plus grande signifie que chaque arbre peut apprendre des relations plus complexes entre les variables d'entrée et les sorties souhaitées, mais cela peut également rendre le modèle plus complexe et augmenter le risque de surapprentissage. Il est donc important de trouver un équilibre entre la profondeur maximale et les performances de prédiction.
# 
# #min_samples_split : C'est le nombre minimal d'échantillons requis pour diviser un nœud. Si le nombre d'échantillons dans un nœud est inférieur à cette valeur, le nœud ne sera pas divisé. Cela peut aider à prévenir le surapprentissage en limitant le nombre de nœuds créés et en favorisant les divisions qui ont une contribution significative à la prédiction.
# 
# #min_samples_leaf : C'est le nombre minimal d'échantillons requis pour être une feuille. Si le nombre d'échantillons dans une feuille est inférieur à cette valeur, la feuille sera fusionnée avec une autre feuille. Cela peut aider à prévenir le surapprentissage en limitant le nombre de feuilles créées et en favorisant les feuilles qui ont une contribution significative à la prédiction.
# 
# #max_features : C'est le nombre maximal de fonctionnalités à considérer lors de la recherche de la meilleure division. Si None, alors toutes les fonctionnalités sont considérées, sinon, une valeur entière doit être passée. Limiter le nombre de fonctionnalités peut aider à réduire la complexité du modèle et à prévenir le surapprentissage.

# ### Faire des prédictions sur l'ensemble de test

# In[109]:


Y8_pred = model2.predict(X_test_selected)
print(Y_test)
print(Y8_pred)


# In[110]:


from sklearn.metrics import accuracy_score


# In[111]:


# Calcul de la précision du modèle
accuracy = accuracy_score(Y_test, Y8_pred)
print("Précision : ", accuracy)


# In[112]:


with np.printoptions(threshold=np.inf):
    print(Y_test)


# In[113]:


with np.printoptions(threshold=np.inf):
    print(Y8_pred)


# In[114]:


from sklearn.metrics import confusion_matrix
CM8 = confusion_matrix(Y_test, Y8_pred)
CM8


# In[115]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                CM8.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     CM8.flatten()/np.sum(CM8)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(CM8, annot=labels, fmt='', cmap='crest')


# In[116]:


print(Y_test.shape)
print(CM8)


# # Courbe roc 

# In[117]:


y_pred_proba8 = model2.predict_proba(X_test_selected)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba8)
auc4 = metrics.roc_auc_score(Y_test, y_pred_proba8)
plt.plot(fpr,tpr,label="df1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# La courbe roc montre la spécifité et la sensibilité


# ### ACCURACY RATE, ERROR RATE & F1-SCORE

# In[118]:


Accuracy_Rate8 = accuracy_score(Y_test, Y8_pred)
Error_rate8 = 1 - Accuracy_Rate8  # Change variable name to Error_rate2
F1_score_logreg8 = f1_score(Y_test, Y8_pred)
precision8 = precision_score(Y_test, Y8_pred)
recall8 = recall_score(Y_test, Y8_pred)
CK8 = cohen_kappa_score(Y_test, Y8_pred)
MC8 = matthews_corrcoef(Y_test, Y8_pred)
auc4 = metrics.roc_auc_score(Y_test, y_pred_proba8)

print("precision : {:.2f}".format(precision8))
print("recall : {:.2f}".format(recall8))
print("Accuracy rate: ", Accuracy_Rate8)
print("Error rate: ", Error_rate8)
print("F1_score: ", F1_score_logreg8)
print("CK:", CK8)
print("AUC:", auc4)


# In[119]:


# create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "Error Rate", "F1 Score", "CK", "MC","AUC"]
metric_values = [precision8, recall8, Accuracy_Rate8, Error_rate8, F1_score_logreg8, CK8, MC8,auc4]

# create a bar chart
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(metric_names, metric_values)
ax.set_ylabel('Value')
ax.set_ylim([0,1])
ax.set_title('Performance Metrics')
plt.show()


# In[120]:


# Données de performance pour chaque modèle
model_data = {
    'RL': {
        'Précision': precision,
        'Rappel': recall,
        'Taux de précision': Accuracy_Rate,
        'Taux d\'erreur': Error_rate,
        'Score F1': F1_score_logreg,
        'CK': CK,
        'MC': MC,
        'AUC': auc
    },
    'SVM': {
        'Précision': precision2,
        'Rappel': recall2,
        'Taux de précision': Accuracy_Rate2,
        'Taux d\'erreur': Error_rate2,
        'Score F1': F1_score_logreg2,
        'CK': CK2,
        'MC': MC2,
        'AUC': auc2
    },
    'RDF': {
        'Précision': precision6,
        'Rappel': recall6,
        'Taux de précision': Accuracy_Rate6,
        'Taux d\'erreur': Error_rate6,
        'Score F1': F1_score_logreg6,
        'CK': CK6,
        'MC': MC6,
        'AUC': auc3
    },
    'GB': {
        'Précision': precision8,
        'Rappel': recall8,
        'Taux de précision': Accuracy_Rate8,
        'Taux d\'erreur': Error_rate8,
        'Score F1': F1_score_logreg8,
        'CK': CK8,
        'MC': MC8,
        'AUC': auc4
    }
}

# Créer une nouvelle figure
plt.figure(figsize=(10, 6))

# Liste des métriques de performance à tracer
performance_metrics = list(model_data['RL'].keys())

# Couleurs pour chaque métrique
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k','#FFA500']

# Tracer chaque métrique de performance pour chaque modèle
for i, metric in enumerate(performance_metrics):
    values = [model_data[model][metric] for model in model_data]
    plt.plot(list(model_data.keys()), values, color=colors[i], label=metric)

# Ajouter une légende et un titre
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Comparaison de modèles')

# Ajuster les limites des axes
plt.ylim([0, 1])
plt.xlim([-0.5, len(model_data)-0.5])

# Nommer les axes
plt.xlabel('Modèles')
plt.ylabel('Score')


# Afficher le graphique
plt.show()


# In[121]:


# Données de performance pour chaque modèle
model_data = {
    'RL': {
        'precision': precision,
        'recall': recall,
        'accuracy': Accuracy_Rate,
        'error': Error_rate,
        'f1_score': F1_score_logreg,
        'CK': CK,
        'MC': MC,
        'AUC': auc
    },
    'SVM': {
        'precision': precision2,
        'recall': recall2,
        'accuracy': Accuracy_Rate2,
        'error': Error_rate2,
        'f1_score': F1_score_logreg2,
        'CK': CK2,
        'MC': MC2,
        'AUC': auc2
    },
    'RDF': {
        'precision': precision6,
        'recall': recall6,
        'accuracy': Accuracy_Rate6,
        'error': Error_rate6,
        'f1_score': F1_score_logreg6,
        'CK': CK6,
        'MC': MC6,
        'AUC': auc3
    },
    'GB': {
        'precision': precision8,
        'recall': recall8,
        'accuracy': Accuracy_Rate8,
        'error': Error_rate8,
        'f1_score': F1_score_logreg8,
        'CK': CK8,
        'MC': MC8,
        'AUC': auc4
    }
}

# Créer un DataFrame pour chaque métrique de performance
precision_df = pd.DataFrame.from_dict({k: v['precision'] for k, v in model_data.items()}, orient='index')
recall_df = pd.DataFrame.from_dict({k: v['recall'] for k, v in model_data.items()}, orient='index')
accuracy_df = pd.DataFrame.from_dict({k: v['accuracy'] for k, v in model_data.items()}, orient='index')
error_df = pd.DataFrame.from_dict({k: v['error'] for k, v in model_data.items()}, orient='index')
f1_score_df = pd.DataFrame.from_dict({k: v['f1_score'] for k, v in model_data.items()}, orient='index')
CK_df = pd.DataFrame.from_dict({k: v['CK'] for k, v in model_data.items()}, orient='index')
MC_df = pd.DataFrame.from_dict({k: v['MC'] for k, v in model_data.items()}, orient='index')
AUC_df = pd.DataFrame.from_dict({k: v['AUC'] for k, v in model_data.items()}, orient='index')

# Placer tous les tableaux dans une liste pour les concaténer horizontalement
dfs = [precision_df, recall_df, accuracy_df, error_df, f1_score_df, CK_df, MC_df, AUC_df]

# Concaténer horizontalement les tableaux pour créer un seul DataFrame
results_df = pd.concat(dfs, axis=1)

# Renommer les colonnes pour qu'elles correspondent aux métriques de performance
results_df.columns = ['precision', 'recall', 'accuracy', 'error', 'f1_score', 'CK', 'MC', 'AUC']

# Tracer les performances de chaque modèle pour chaque métrique
results_df.plot(kind='bar',figsize=(15, 7))
plt.title('Comparaison de modèle')
plt.ylabel('Score')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# # DEEP LEARNING

# ### Artificial Neural Network (ANN)

# Le réseau de neurones artificiels (ANN) est un modèle d'apprentissage en profondeur (Deep Learning) qui vise à imiter le fonctionnement du cerveau humain pour résoudre des problèmes complexes tels que la reconnaissance de motifs, la classification et la prédiction. Le modèle est constitué de plusieurs couches de neurones artificiels, chacune avec une fonction d'activation qui permet au modèle de capturer des relations non linéaires dans les données.
# 
# L'ANN est généralement entraîné à l'aide de la méthode de la rétropropagation (backpropagation), qui ajuste les poids de chaque neurone pour minimiser la fonction de coût. Le modèle peut être utilisé pour la classification et la régression, et peut gérer des données structurées et non structurées telles que des images, des séquences de texte et des séries chronologiques.
# 
# Les avantages de l'ANN comprennent sa capacité à apprendre à partir de grandes quantités de données et à capturer des relations complexes dans les données. Cependant, l'entraînement du modèle peut être coûteux en termes de temps de calcul et de ressources de calcul nécessaires, et il peut être difficile d'interpréter les résultats du modèle.

# In[122]:


import keras
from keras.models import Sequential # C'est le module qui va nous permettre d'initialiser le reseau de neuronne 
from keras.layers import Dense # C'est le module qui nous permet de créer les couches des réseaux de neuronne 
from keras.layers import Dropout


# In[123]:


# Initialisation du réseau de neurones
classifier = Sequential()

# Ajout de la couche d'entrée et de la première couche cachée
classifier.add(Dense(units=65, activation='relu', input_dim=63,))
classifier.add(Dropout(0.2))

# Ajout de la deuxième couche cachée
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dropout(0.2))

# Ajout de la troisième couche cachée
classifier.add(Dense(units=16, activation='relu'))
classifier.add(Dropout(0.2))

# Ajout de la couche de sortie
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilation du réseau de neurones
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du réseau de neurones
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)


# In[124]:


# Évaluer le modèle sur les données de test
loss, accuracy = classifier.evaluate(X_test, Y_test)
print("Perte (loss) sur les données de test: ", loss)
print("Accuracy sur les données de test: ", accuracy)


# In[125]:


# Evaluation de l'accuracy rate
Y_pred_9 = classifier.predict(X_test)
Accuracy_Rate9 = accuracy_score(Y_test, Y_pred_9.round())
print("Accuracy rate:", accuracy)

# Evaluation de l'error rate
Error_rate9 = 1 - accuracy
print("Error rate:", Error_rate9)

# Evaluation du F1-score
F1_score_ANN9 = f1_score(Y_test, Y_pred_9.round())
print("F1-score:", F1_score_ANN9)


# In[126]:


precision9 = precision_score(Y_test, Y_pred_9.round())
recall9 = recall_score(Y_test,  Y_pred_9.round())
CK9 = cohen_kappa_score(Y_test,  Y_pred_9.round())
MC9 = matthews_corrcoef(Y_test, Y_pred_9.round())
auc5 = metrics.roc_auc_score(Y_test, Y_pred_9.round())


# In[127]:


# create a list of metric names and values
metric_names = ["Precision", "Recall", "Accuracy Rate", "Error Rate", "F1 Score", "CK", "MC","AUC"]
metric_values = [precision9, recall9, Accuracy_Rate9, Error_rate9, F1_score_ANN9, CK9, MC9,auc]

# create a bar chart
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(metric_names, metric_values)
ax.set_ylabel('Value')
ax.set_ylim([0,1])
ax.set_title('Performance Metrics')
plt.show()


# In[128]:


# Données de performance pour chaque modèle
model_data = {
    'RL': {
        'precision': precision,
        'recall': recall,
        'accuracy': Accuracy_Rate,
        'error': Error_rate,
        'f1_score': F1_score_logreg,
        'CK': CK,
        'MC': MC,
        'AUC': auc
    },
    'SVM': {
        'precision': precision2,
        'recall': recall2,
        'accuracy': Accuracy_Rate2,
        'error': Error_rate2,
        'f1_score': F1_score_logreg2,
        'CK': CK2,
        'MC': MC2,
        'AUC': auc2
    },
    'RF': {
        'precision': precision6,
        'recall': recall6,
        'accuracy': Accuracy_Rate6,
        'error': Error_rate6,
        'f1_score': F1_score_logreg6,
        'CK': CK6,
        'MC': MC6,
        'AUC': auc3
    },
    'GB': {
        'precision': precision8,
        'recall': recall8,
        'accuracy': Accuracy_Rate8,
        'error': Error_rate8,
        'f1_score': F1_score_logreg8,
        'CK': CK8,
        'MC': MC8,
        'AUC': auc4
    },
    'ANN': {
        'precision': precision9,
        'recall': recall9,
        'accuracy': Accuracy_Rate9,
        'error': Error_rate9,
        'f1_score': F1_score_ANN9 ,
        'CK': CK9,
        'MC': MC9,
        'AUC': auc5
}
                }

# Créer un DataFrame pour chaque métrique de performance
precision_df = pd.DataFrame.from_dict({k: v['precision'] for k, v in model_data.items()}, orient='index')
recall_df = pd.DataFrame.from_dict({k: v['recall'] for k, v in model_data.items()}, orient='index')
accuracy_df = pd.DataFrame.from_dict({k: v['accuracy'] for k, v in model_data.items()}, orient='index')
error_df = pd.DataFrame.from_dict({k: v['error'] for k, v in model_data.items()}, orient='index')
f1_score_df = pd.DataFrame.from_dict({k: v['f1_score'] for k, v in model_data.items()}, orient='index')
CK_df = pd.DataFrame.from_dict({k: v['CK'] for k, v in model_data.items()}, orient='index')
MC_df = pd.DataFrame.from_dict({k: v['MC'] for k, v in model_data.items()}, orient='index')
AUC_df = pd.DataFrame.from_dict({k: v['AUC'] for k, v in model_data.items()}, orient='index')

# Placer tous les tableaux dans une liste pour les concaténer horizontalement
dfs = [precision_df, recall_df, accuracy_df, error_df, f1_score_df, CK_df, MC_df,AUC_df]

# Concaténer horizontalement les tableaux pour créer un seul DataFrame
results_df = pd.concat(dfs, axis=1)

# Renommer les colonnes pour qu'elles correspondent aux métriques de performance
results_df.columns = ['precision', 'recall', 'accuracy', 'error', 'f1_score', 'CK', 'MC','AUC']

# Afficher le DataFrame
print(results_df)
# Tracer les performances de chaque modèle pour chaque métrique
results_df.plot(kind='bar',figsize=(15, 7))
plt.title('Comparaison de modèle')
plt.ylabel('Score')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# ### Les variables issuent de la  feature selection

# In[129]:


Var_feature_selection = df[['SEX_M', 'ENFANT_BIN', 'PHARMACIEN', 'SAGEFEMME',
       'SANITAIRE_VS_MEDICOSOCIAL', '86SUIVI_PSYCHO', 'DETA_CUTOFF2',
       'KARASEK_AXE1_DEMANDE_PSYCHOLOGIQUE', 'KARASEK_AXE3_SOUTIEN_SOCIAL',
       'KARASEK_AXE2_LATITUDE_DECISIONELLE', '75CRAINTE_ERREUR_FQC',
       'PSQI_QUALITE_CONTINU', 'BURNOUT_BIN',
       '127. Au cours des 30 derniers jours, combien vous a-t-il été difficile d’être suffisamment motivé(e) pour mener à bien vos activités ?']]


# In[130]:


Var_feature_selection.shape


# In[131]:


# Sélectionner les variables binaires

binary_vars = Var_feature_selection[['SEX_M', 'ENFANT_BIN', 'PHARMACIEN', 'SAGEFEMME',
       'SANITAIRE_VS_MEDICOSOCIAL', '86SUIVI_PSYCHO', 'DETA_CUTOFF2']]

# Sélectionner les variables continues
continuous_vars = Var_feature_selection[['KARASEK_AXE1_DEMANDE_PSYCHOLOGIQUE', 'KARASEK_AXE3_SOUTIEN_SOCIAL',
       'KARASEK_AXE2_LATITUDE_DECISIONELLE', '75CRAINTE_ERREUR_FQC',
       'PSQI_QUALITE_CONTINU',
       '127. Au cours des 30 derniers jours, combien vous a-t-il été difficile d’être suffisamment motivé(e) pour mener à bien vos activités ?']]


# In[132]:


#Affichage 
print(continuous_vars)


# ### Test de chi2 pour les variables binaires

# In[133]:


from scipy.stats import chi2_contingency

# Supposons que 'df' est le nom de votre dataframe et que vous voulez tester les variables binaires dans la liste 'binary_vars'
for var in binary_vars:
    contingency_table = pd.crosstab(Var_feature_selection[var], Var_feature_selection['BURNOUT_BIN'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Variable {var}: chi2 = {chi2:.3f}, p = {p:.3f}")


# ### Test de Student

# In[134]:


from scipy.stats import ttest_ind

# Faire un test de Student pour chaque variable continue
for var in continuous_vars.columns:
    group1 = Var_feature_selection[Var_feature_selection['BURNOUT_BIN'] == 0][var]
    group2 = Var_feature_selection[Var_feature_selection['BURNOUT_BIN'] == 1][var]
    t, p = ttest_ind(group1, group2)
    print(f"Variable {var}: t = {t:.3f}, p = {p:.3f}")


# In[135]:


df.shape


# In[136]:


df.columns


# # To do 

# In[137]:


Var_feature_selection_xi = df[['SEX_M', 'ENFANT_BIN', 'PHARMACIEN', 'SAGEFEMME',
       'SANITAIRE_VS_MEDICOSOCIAL', '86SUIVI_PSYCHO', 'DETA_CUTOFF2',
       'KARASEK_AXE1_DEMANDE_PSYCHOLOGIQUE', 'KARASEK_AXE3_SOUTIEN_SOCIAL',
       'KARASEK_AXE2_LATITUDE_DECISIONELLE', '75CRAINTE_ERREUR_FQC',
       'PSQI_QUALITE_CONTINU','127. Au cours des 30 derniers jours, combien vous a-t-il été difficile d’être suffisamment motivé(e) pour mener à bien vos activités ?']]

