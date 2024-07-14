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


# # Données manquantes 

# In[30]:


df1.isnull().sum()[df1.isnull().sum()>0]


# # Gestion des variables binaires et non binaires

# In[31]:


binary_vars = []
non_binary_vars = []
for col in df1.columns:
    if df1[col].nunique() == 2:
        binary_vars.append(col)
    else:
        non_binary_vars.append(col)


# In[32]:


# Afficher les colonnes binaires
print('Colonnes binaires :', binary_vars)


# In[33]:


# Afficher les colonnes non_binaires
print('Colonnes non binaires :', non_binary_vars)


# In[34]:


print(len(non_binary_vars))
print(len(binary_vars))


# In[35]:


binary_vars


# In[36]:


binary_vars


# # Variables binaires

# In[37]:


# Affichage des valeurs uniques et comptage dans chaque colonne de binary_vars
for col in binary_vars:
    unique_values = df1[col].unique()
    count_values = df1[col].value_counts()
    print(f"Valeurs uniques dans {col} : {unique_values}")
    print(f"Nombre de variables dans {col} : {len(count_values)}\n")


# # Vérification du pourcentage des hommes
# 
# ### ['SEX_M'] == 1 correspond aux hommes 
# ### ['SEX_M'] == 0  correspond aux femmes 

# In[38]:


# Compter le nombre total d'individus
total_individuals = len(df)

# Compter le nombre d'hommes
men_count = len(df[df['SEX_M'] == 1])

# Calculer le pourcentage des hommes
percentage_men = (men_count / total_individuals) * 100

# Afficher le résultat
print("Pourcentage d'hommes : {:.2f}%".format(percentage_men))


# In[39]:


from tabulate import tabulate

# Création d'une liste pour stocker les résultats
result_list = []

# Boucle pour chaque variable binaire
for col in binary_vars:
    # Comptage du nombre de BURNOUT_BIN = 0 et BURNOUT_BIN = 1 pour la variable binaire
    count_0 = df1[df1['BURNOUT_BIN'] == 0][col].sum()
    count_1 = df1[df1['BURNOUT_BIN'] == 1][col].sum()

    # Ajout des résultats à la liste
    result_list.append([col, count_0, count_1])

# Affichage du tableau tabulé avec une mise en forme personnalisée
table = tabulate(result_list, headers=['Variable', 'BURNOUT_BIN = 0', 'BURNOUT_BIN = 1'], tablefmt='psql')

# Affichage du tableau
print(table)


# In[41]:


import pandas as pd
from scipy.stats import chi2_contingency
from tabulate import tabulate

# Sélection des variables binaires
binary_vars = [col for col in df1.columns if df1[col].nunique() == 2]

# Création d'une liste pour stocker les résultats
result_list = []

# Boucle pour chaque variable binaire
for col in binary_vars:
    # Nombre de BURNOUT_BIN = 0 et BURNOUT_BIN = 1 pour les hommes
    counts_0_men = df1[(df1['BURNOUT_BIN'] == 0) & (df1['SEX_M'] == 0)][col].sum()
    counts_1_men = df1[(df1['BURNOUT_BIN'] == 1) & (df1['SEX_M'] == 0)][col].sum()

    # Nombre de valeurs manquantes pour les hommes
    missing_count_men = df1[df1['SEX_M'] == 0][col].isnull().sum()

    # Test du chi2 pour les hommes
    contingency_table_men = pd.crosstab(df1[df1['SEX_M'] == 0][col], df1[df1['SEX_M'] == 0]['BURNOUT_BIN'])
    chi2_men, p_value_men, _, _ = chi2_contingency(contingency_table_men)

    # Vérification de la condition p < 0.05 pour les hommes
    p_value_status_men = "Significant" if p_value_men < 0.05 else "Not Significant"

    # Ajout des résultats pour les hommes à la liste
    result_list.append(['Hommes', col, counts_0_men, counts_1_men, missing_count_men, chi2_men, p_value_men, p_value_status_men])

    # Nombre de BURNOUT_BIN = 0 et BURNOUT_BIN = 1 pour les femmes
    counts_0_women = df1[(df1['BURNOUT_BIN'] == 0) & (df1['SEX_M'] == 1)][col].sum()
    counts_1_women = df1[(df1['BURNOUT_BIN'] == 1) & (df1['SEX_M'] == 1)][col].sum()

    # Nombre de valeurs manquantes pour les femmes
    missing_count_women = df1[df1['SEX_M'] == 1][col].isnull().sum()

    # Test du chi2 pour les femmes
    contingency_table_women = pd.crosstab(df1[df1['SEX_M'] == 1][col], df1[df1['SEX_M'] == 1]['BURNOUT_BIN'])
    chi2_women, p_value_women, _, _ = chi2_contingency(contingency_table_women)

    # Vérification de la condition p < 0.05 pour les femmes
    p_value_status_women = "Significant" if p_value_women < 0.05 else "Not Significant"

    # Ajout des résultats pour les femmes à la liste
    result_list.append(['Femmes', col, counts_0_women, counts_1_women, missing_count_women, chi2_women, p_value_women, p_value_status_women])

# Conversion de la liste en DataFrame
result_df = pd.DataFrame(result_list, columns=['Groupe', 'Variable', 'BURNOUT_BIN = 0 (Count)', 'BURNOUT_BIN = 1 (Count)', 'Missing Count', 'Chi2', 'P-value', 'P-value Status'])

# Affichage du tableau tabulé avec une mise en forme différente
print("Tableau des informations pour chaque variable binaire :\n")
print(tabulate(result_df, headers='keys', tablefmt='grid', showindex=False))


# ### Pourcentage de Y = 0 et de Y= 1 et Effet de size 

# In[60]:


import pandas as pd
from scipy.stats import chi2_contingency
from tabulate import tabulate
from math import sqrt

## Sélection des variables binaires
binary_vars = [col for col in df1.columns if df1[col].nunique() == 2]

## Création d'une liste pour stocker les résultats
result_list = []
significant_vars = []

## Boucle pour chaque variable binaire
for col in binary_vars:
    # Pourcentage de BURNOUT_BIN = 0 et BURNOUT_BIN = 1 pour les hommes
    total_men = df1[df1['SEX_M'] == 0][col].count()
    percent_0_men = (df1[(df1['BURNOUT_BIN'] == 0) & (df1['SEX_M'] == 0)][col].sum() / total_men) * 100
    percent_1_men = (df1[(df1['BURNOUT_BIN'] == 1) & (df1['SEX_M'] == 0)][col].sum() / total_men) * 100

    # Nombre de valeurs manquantes pour les hommes
    missing_count_men = df1[df1['SEX_M'] == 0][col].isnull().sum()

    # Test du chi2 pour les hommes
    contingency_table_men = pd.crosstab(df1[df1['SEX_M'] == 0][col], df1[df1['SEX_M'] == 0]['BURNOUT_BIN'])
    chi2_men, p_value_men, _, _ = chi2_contingency(contingency_table_men)

    # Vérification de la condition p < 0.05 pour les hommes
    p_value_status_men = "Significant" if p_value_men < 0.05 else "Not Significant"

    # Effect size pour les hommes
    n_0_men = df1[(df1['BURNOUT_BIN'] == 0) & (df1['SEX_M'] == 0)][col].sum()
    n_1_men = df1[(df1['BURNOUT_BIN'] == 1) & (df1['SEX_M'] == 0)][col].sum()
    effect_size_men = (n_1_men / total_men - n_0_men / total_men) / sqrt((n_1_men + n_0_men) / total_men)

    # Ajout des résultats pour les hommes à la liste
    result_list.append(['Hommes', col, f"{percent_0_men:.2f}%", f"{percent_1_men:.2f}%", missing_count_men, chi2_men, p_value_men, p_value_status_men, effect_size_men])

    # Pourcentage de BURNOUT_BIN = 0 et BURNOUT_BIN = 1 pour les femmes
    total_women = df1[df1['SEX_M'] == 1][col].count()
    percent_0_women = (df1[(df1['BURNOUT_BIN'] == 0) & (df1['SEX_M'] == 1)][col].sum() / total_women) * 100
    percent_1_women = (df1[(df1['BURNOUT_BIN'] == 1) & (df1['SEX_M'] == 1)][col].sum() / total_women) * 100

    # Nombre de valeurs manquantes pour les femmes
    missing_count_women = df1[df1['SEX_M'] == 1][col].isnull().sum()

    # Test du chi2 pour les femmes
    contingency_table_women = pd.crosstab(df1[df1['SEX_M'] == 1][col], df1[df1['SEX_M'] == 1]['BURNOUT_BIN'])
    chi2_women, p_value_women, _, _ = chi2_contingency(contingency_table_women)

    # Vérification de la condition p < 0.05 pour les femmes
    p_value_status_women = "Significant" if p_value_women < 0.05 else "Not Significant"

    # Effect size pour les femmes
    n_0_women = df1[(df1['BURNOUT_BIN'] == 0) & (df1['SEX_M'] == 1)][col].sum()
    n_1_women = df1[(df1['BURNOUT_BIN'] == 1) & (df1['SEX_M'] == 1)][col].sum()
    effect_size_women = (n_1_women / total_women - n_0_women / total_women) / sqrt((n_1_women + n_0_women) / total_women)

    # Ajout des résultats pour les femmes à la liste
    result_list.append(['Femmes', col, f"{percent_0_women:.2f}%", f"{percent_1_women:.2f}%", missing_count_women, chi2_women, p_value_women, p_value_status_women, effect_size_women])

    # Vérification de la condition p < 0.05 pour les variables significatives
    if p_value_men < 0.05 or p_value_women < 0.05:
        significant_vars.append(col)

## Conversion de la liste en DataFrame
result_df = pd.DataFrame(result_list, columns=['Groupe', 'Variable', 'BURNOUT_BIN = 0 (%)', 'BURNOUT_BIN = 1 (%)', 'Missing Count', 'Chi2', 'P-value', 'P-value Status', 'Effect Size'])

## Enregistrement du DataFrame complet dans un fichier Excel
with pd.ExcelWriter('AMADEUS_Tableau_Statistique_Variables_binaires.xlsx') as writer:
    result_df.to_excel(writer, sheet_name='Toutes les Variables', index=False)

    ## Création d'un DataFrame pour les variables significatives
    significant_df = result_df[result_df['Variable'].isin(significant_vars)]
    ## Enregistrement du DataFrame des variables significatives dans un autre classeur Excel
    significant_df.to_excel(writer, sheet_name='Variables Significatives', index=False)

## Affichage du tableau tabulé avec une mise en forme différente
print("Tableau des informations pour chaque variable binaire :\n")
print(tabulate(result_df, headers='keys', tablefmt='grid', showindex=False))


# ### Effect size
# 
# L'effect size est une mesure statistique qui quantifie l'importance ou la magnitude d'une différence ou d'une association entre deux groupes ou variables. Il est utilisé pour évaluer l'ampleur de l'effet observé, indépendamment de la taille de l'échantillon.
# 
# Dans le contexte du code fourni, l'effect size est calculé pour chaque variable binaire en utilisant la formule suivante :
# 
# 
# effect_size = (n_1 / total - n_0 / total) / sqrt((n_1 + n_0) / total)
# où :
# 
# n_1 est le nombre d'observations avec la valeur BURNOUT_BIN = 1,
# n_0 est le nombre d'observations avec la valeur BURNOUT_BIN = 0,
# total est le nombre total d'observations dans le groupe considéré (hommes ou femmes),
# sqrt représente la fonction racine carrée.
# L'effect size ainsi calculé est standardisé et exprimé en écarts-types. Il mesure la différence normalisée entre les proportions de BURNOUT_BIN = 1 et BURNOUT_BIN = 0, ajustée en fonction de la taille de l'échantillon.
# 
# L'interprétation de l'effect size dépend du contexte spécifique de l'étude et de la nature de la variable binaire. En général, un effect size plus élevé indique une différence plus importante ou une association plus forte entre les variables. Cependant, il n'y a pas de règle universelle pour déterminer ce qui est considéré comme un effect size "important" ou "cliniquement significatif". Cela dépend du domaine d'étude, de la portée de la recherche et des connaissances préalables dans le domaine.
# 
# Lorsque l'effect size est positif, cela indique une probabilité plus élevée d'observation de BURNOUT_BIN = 1 par rapport à BURNOUT_BIN = 0. Lorsque l'effect size est négatif, cela indique une probabilité plus faible d'observation de BURNOUT_BIN = 1 par rapport à BURNOUT_BIN = 0.
# 
# Il est important de considérer l'effect size en conjonction avec d'autres mesures statistiques telles que le test du chi2 et la valeur de p. Ces mesures combinées fournissent une compréhension plus complète de la relation entre les variables et aident à évaluer leur pertinence et leur importance dans le contexte de l'étude.
# 
# Il convient également de noter que l'interprétation de l'effect size peut varier en fonction du domaine d'étude et des conventions spécifiques à ce domaine. Il est donc conseillé de consulter la littérature pertinente et de prendre en compte les recommandations spécifiques à votre domaine lors de l'interprétation de l'effect size.

# # Variable non binaires ou continues 

# In[42]:


from tabulate import tabulate

# Création d'une liste pour stocker les résultats
result_list = []

# Boucle pour chaque variable non binaire
for col in non_binary_vars:
    # Moyenne et écart type pour BURNOUT_BIN = 0
    mean_0 = df1[df1['BURNOUT_BIN'] == 0][col].mean()
    std_0 = df1[df1['BURNOUT_BIN'] == 0][col].std()

    # Moyenne et écart type pour BURNOUT_BIN = 1
    mean_1 = df1[df1['BURNOUT_BIN'] == 1][col].mean()
    std_1 = df1[df1['BURNOUT_BIN'] == 1][col].std()

    # Ajout des résultats à la liste
    result_list.append([col, f"{mean_0:.2f} ± {std_0:.2f}", f"{mean_1:.2f} ± {std_1:.2f}"])

# Affichage du tableau tabulé avec une mise en forme personnalisée
table = tabulate(result_list, headers=['Variable', 'BURNOUT_BIN = 0', 'BURNOUT_BIN = 1'], tablefmt='psql')

# Affichage du tableau
print(table)


# In[43]:


from tabulate import tabulate
from scipy.stats import ttest_ind

# Création d'une liste pour stocker les résultats
result_list = []

# Boucle pour chaque variable non binaire
for col in non_binary_vars:
    # Test de Student
    t_stat, p_value = ttest_ind(df1[df1['BURNOUT_BIN'] == 0][col].dropna(), df1[df1['BURNOUT_BIN'] == 1][col].dropna(), equal_var=False)

    # Ajout des résultats à la liste
    result_list.append([col, f"Test de Student (t-statistique): {t_stat:.2f}", f"p-value: {p_value:.2f}"])

# Affichage du tableau tabulé avec une mise en forme personnalisée
table = tabulate(result_list, headers=['Variable', 'Test de Student (t-statistique)', 'p-value'], tablefmt='psql')

# Affichage du tableau
print(table)


# In[44]:


from tabulate import tabulate

# Création d'une liste pour stocker les résultats
result_list = []

# Boucle pour chaque variable non binaire
for col in non_binary_vars:
    # Calcul du nombre de valeurs manquantes pour BURNOUT_BIN = 0
    missing_0 = df1[df1['BURNOUT_BIN'] == 0][col].isnull().sum()

    # Calcul du nombre de valeurs manquantes pour BURNOUT_BIN = 1
    missing_1 = df1[df1['BURNOUT_BIN'] == 1][col].isnull().sum()

    # Ajout des résultats à la liste
    result_list.append([col, missing_0, missing_1])

# Affichage du tableau tabulé avec une mise en forme personnalisée
table = tabulate(result_list, headers=['Variable', 'Valeurs manquantes (BURNOUT_BIN = 0)', 'Valeurs manquantes (BURNOUT_BIN = 1)'], tablefmt='psql')

# Affichage du tableau
print(table)


# In[45]:


import pandas as pd
from scipy.stats import ttest_ind
from tabulate import tabulate

# Création d'une liste pour stocker les résultats
result_list = []

# Boucle pour chaque variable non binaire
for col in non_binary_vars:
    # Calcul de la moyenne
    mean_0 = df1[df1['BURNOUT_BIN'] == 0][col].mean()
    mean_1 = df1[df1['BURNOUT_BIN'] == 1][col].mean()

    # Calcul de l'écart-type
    std_0 = df1[df1['BURNOUT_BIN'] == 0][col].std()
    std_1 = df1[df1['BURNOUT_BIN'] == 1][col].std()

    # Calcul du nombre de valeurs manquantes
    missing_0 = df1[df1['BURNOUT_BIN'] == 0][col].isnull().sum()
    missing_1 = df1[df1['BURNOUT_BIN'] == 1][col].isnull().sum()

    # Test de Student
    t_stat, p_value = ttest_ind(df1[df1['BURNOUT_BIN'] == 0][col].dropna(), df1[df1['BURNOUT_BIN'] == 1][col].dropna(), equal_var=False)

    # Ajout des résultats à la liste
    result_list.append([col, f"{mean_0:.2f}", f"{mean_1:.2f}", f"{std_0:.2f}", f"{std_1:.2f}", missing_0, missing_1, t_stat, p_value])

# Création du DataFrame à partir de la liste de résultats
result_df = pd.DataFrame(result_list, columns=['Variable', 'Moyenne (BURNOUT_BIN = 0)', 'Moyenne (BURNOUT_BIN = 1)', 'Écart-type (BURNOUT_BIN = 0)', 'Écart-type (BURNOUT_BIN = 1)', 'Valeurs manquantes (BURNOUT_BIN = 0)', 'Valeurs manquantes (BURNOUT_BIN = 1)', 'Test de Student (t-statistique)', 'p-value'])

# Enregistrement du DataFrame dans un fichier Excel
result_df.to_excel('C:/Users/aliah/resultats_ENFIN.xlsx', index=False)


# In[46]:


import pandas as pd
from scipy.stats import ttest_ind
from tabulate import tabulate

# Création d'une liste pour stocker les résultats
result_list = []
significant_vars = []

# Boucle pour chaque variable non binaire
for col in non_binary_vars:
    # Calcul de la moyenne
    mean_0 = df1[df1['BURNOUT_BIN'] == 0][col].mean()
    mean_1 = df1[df1['BURNOUT_BIN'] == 1][col].mean()

    # Calcul de l'écart-type
    std_0 = df1[df1['BURNOUT_BIN'] == 0][col].std()
    std_1 = df1[df1['BURNOUT_BIN'] == 1][col].std()

    # Calcul du nombre de valeurs manquantes
    missing_0 = df1[df1['BURNOUT_BIN'] == 0][col].isnull().sum()
    missing_1 = df1[df1['BURNOUT_BIN'] == 1][col].isnull().sum()

    # Test de Student
    t_stat, p_value = ttest_ind(df1[df1['BURNOUT_BIN'] == 0][col].dropna(), df1[df1['BURNOUT_BIN'] == 1][col].dropna(), equal_var=False)

    # Ajout des résultats à la liste
    result_list.append([col, f"{mean_0:.2f}", f"{mean_1:.2f}", f"{std_0:.2f}", f"{std_1:.2f}", missing_0, missing_1, t_stat, p_value])

    # Vérification de la significativité
    if p_value < 0.05:
        significant_vars.append(col)

# Création du DataFrame à partir de la liste de résultats
result_df = pd.DataFrame(result_list, columns=['Variable', 'Moyenne (BURNOUT_BIN = 0)', 'Moyenne (BURNOUT_BIN = 1)', 'Écart-type (BURNOUT_BIN = 0)', 'Écart-type (BURNOUT_BIN = 1)', 'Valeurs manquantes (BURNOUT_BIN = 0)', 'Valeurs manquantes (BURNOUT_BIN = 1)', 'Test de Student (t-statistique)', 'p-value'])

# Enregistrement du DataFrame dans un fichier Excel
result_df.to_excel('C:/Users/aliah/resultats_ENFIN.xlsx', index=False)

# Création du DataFrame des variables significatives
significant_df = result_df[result_df['Variable'].isin(significant_vars)]

# Enregistrement du DataFrame des variables significatives dans un fichier Excel
significant_df.to_excel('C:/Users/aliah/AMADEUS_variables_significatives.xlsx', index=False)

# Affichage des résultats
print("Résultats des variables continues :")
print(tabulate(result_df, headers='keys', tablefmt='psql'))
print("\nVariables significatives :")
print(tabulate(significant_df, headers='keys', tablefmt='psql'))


# ### Compléter le code avec les medianes et les intervalles inter-quartiles pour les variables continues 

# In[47]:


import pandas as pd
from scipy.stats import ttest_ind
from tabulate import tabulate

# Création d'une liste pour stocker les résultats
result_list = []
significant_vars = []

# Boucle pour chaque variable non binaire
for col in non_binary_vars:
    # Calcul de la moyenne
    mean_0 = df1[df1['BURNOUT_BIN'] == 0][col].mean()
    mean_1 = df1[df1['BURNOUT_BIN'] == 1][col].mean()

    # Calcul de la médiane
    median_0 = df1[df1['BURNOUT_BIN'] == 0][col].median()
    median_1 = df1[df1['BURNOUT_BIN'] == 1][col].median()

    # Calcul de l'écart-type
    std_0 = df1[df1['BURNOUT_BIN'] == 0][col].std()
    std_1 = df1[df1['BURNOUT_BIN'] == 1][col].std()

    # Calcul des quartiles
    q1_0 = df1[df1['BURNOUT_BIN'] == 0][col].quantile(0.25)
    q3_0 = df1[df1['BURNOUT_BIN'] == 0][col].quantile(0.75)
    q1_1 = df1[df1['BURNOUT_BIN'] == 1][col].quantile(0.25)
    q3_1 = df1[df1['BURNOUT_BIN'] == 1][col].quantile(0.75)

    # Calcul du nombre de valeurs manquantes
    missing_0 = df1[df1['BURNOUT_BIN'] == 0][col].isnull().sum()
    missing_1 = df1[df1['BURNOUT_BIN'] == 1][col].isnull().sum()

    # Test de Student
    t_stat, p_value = ttest_ind(df1[df1['BURNOUT_BIN'] == 0][col].dropna(), df1[df1['BURNOUT_BIN'] == 1][col].dropna(), equal_var=False)

    # Ajout des résultats à la liste
    result_list.append([col, f"{mean_0:.2f}", f"{mean_1:.2f}", f"{median_0:.2f}", f"{median_1:.2f}", f"{std_0:.2f}", f"{std_1:.2f}", f"{q1_0:.2f} - {q3_0:.2f}", f"{q1_1:.2f} - {q3_1:.2f}", missing_0, missing_1, t_stat, p_value])

    # Vérification de la significativité
    if p_value < 0.05:
        significant_vars.append(col)

# Création du DataFrame à partir de la liste de résultats
result_df = pd.DataFrame(result_list, columns=['Variable', 'Moyenne (BURNOUT_BIN = 0)', 'Moyenne (BURNOUT_BIN = 1)', 'Médiane (BURNOUT_BIN = 0)', 'Médiane (BURNOUT_BIN = 1)', 'Écart-type (BURNOUT_BIN = 0)', 'Écart-type (BURNOUT_BIN = 1)', 'Intervalles interquartiles (BURNOUT_BIN = 0)', 'Intervalles interquartiles (BURNOUT_BIN = 1)', 'Valeurs manquantes (BURNOUT_BIN = 0)', 'Valeurs manquantes (BURNOUT_BIN = 1)', 'Test de Student (t-statistique)', 'p-value'])

# Enregistrement du DataFrame dans un fichier Excel
result_df.to_excel('Tableau Statistiques variables continues.xlsx', index=False)

# Création du DataFrame des variables significatives
significant_df = result_df[result_df['Variable'].isin(significant_vars)]

# Enregistrement du DataFrame des variables significatives dans un fichier Excel
significant_df.to_excel('Variables significatives.xlsx', index=False)

# Affichage des résultats
print("Résultats des variables continues :")
print(tabulate(result_df, headers='keys', tablefmt='psql'))
print("\nVariables significatives :")
print(tabulate(significant_df, headers='keys', tablefmt='psql'))


# ### Commentaire sur la manière de calculer les IIQ : 
#        
# L'intervalle interquartile est une mesure de dispersion qui représente la différence entre le troisième quartile (Q3) et le premier quartile (Q1) d'une distribution. Pour calculer l'intervalle interquartile pour chaque variable dans notre cas, nous avons utilisé la fonction quantile de pandas.
# 
# Voici comment le calcul de l'intervalle interquartile a été effectué :
# 
# Pour la classe BURNOUT_BIN = 0, nous avons extrait les valeurs de la variable correspondante (col) pour cette classe à l'aide de la condition df1[df1['BURNOUT_BIN'] == 0][col].
# 
# Ensuite, nous avons utilisé la fonction quantile pour calculer le premier quartile (Q1_0) en utilisant df1[df1['BURNOUT_BIN'] == 0][col].quantile(0.25) et le troisième quartile (Q3_0) en utilisant df1[df1['BURNOUT_BIN'] == 0][col].quantile(0.75).
# 
# Les mêmes étapes ont été répétées pour la classe BURNOUT_BIN = 1, en utilisant les conditions correspondantes et en calculant les quartiles Q1_1 et Q3_1 pour la classe.
# 
# Enfin, nous avons enregistré l'intervalle interquartile comme une chaîne de caractères dans la liste result_list en utilisant le format "{q1_0:.2f} - {q3_0:.2f}" pour la classe BURNOUT_BIN = 0, et de la même manière pour la classe BURNOUT_BIN = 1.

# ### Le % de  y = 0 et y =1 

# In[48]:


percentage_0 = (df['BURNOUT_BIN'] == 0).mean() * 100
percentage_1 = (df['BURNOUT_BIN'] == 1).mean() * 100

print(f"Pourcentage de 0 dans BURNOUT_BIN : {percentage_0:.2f}%")
print(f"Pourcentage de 1 dans BURNOUT_BIN : {percentage_1:.2f}%")

