# Churn Contrats Auto

```
PROJET EN COURS
```

## 📌 Description du projet
Ce projet vise à développer un modèle de scoring ciblant les clients qui vont résilier leur contrat auto. 

## 📁 Arborescence du projet
```
📂 credit_scoring  
│── 📂 data                  # 📊 Données brutes, intermédiaires et traitées  
│   ├── raw                  # 📥 Données brutes (non modifiées)  
│   └── processed            # 🔄 Données pré-traitées  
│  
│── 📂 notebooks             # 📒 Jupyter Notebooks pour exploration et prototypage
│   └── 📂 student_version 
│       ├── 01_Data_exploration.ipynb  
│       ├── 02_Preprocessing.ipynb  
│       └── 03_Modeling.ipynb  
│  
│── 📂 src                   # 💻 Code source du projet  
│   └── 📂 utils   # 🛠️ Fonctions utilisées dans les Notebooks  
│       ├── tools.py
│       └── class_modeling.py
│  
│── .gitignore               # 🚫 Fichiers à exclure du contrôle de version  
│── requirements.txt         # 📦 Dépendances du projet  
│── README.md                # 📝 Présentation du projet  
```

## 📊 Jeu de données
Le jeu de données utilisé provient des données clients d'un assureur auto. Ce dernier contient des informations personnelles sur l'assuré (Sexe, age, region, situation familiale), ainsi que des informations clients (ancienneté client, nombre de contrats, niveau bonus/malus).

### 📌 Description des variables

| 🏷️ **Nom** | 📝 **Description** |
|----------------|-----------------------------|
| **ANCCLI** | Ancienneté du client (date du premier contrat) |
| **AU4R** | Nb de contrats actifs 4 roues |
| **CDMARVEH** | Marque de la voiture |
| **CDMCE** | Code marché professionnel/particulier |
| **CDSITFAM** | Situation familiale |
| **DUSGAUT** | Code d'usage de la voiture |
| **CD_AGT** | Agent qui a vendu le contrat |
| **CD_CSP** | CSP |
| **CD_FML** | Code formule, tout risque, routière… |
| **CD_SEX** | Sexe |
| **CONTRAT** | Résilié / non résilié |
| **DEPT** | Département |
| **DI** | Nombre de contrats divers actifs |
| **DTDBUCON** | Date de début du contrat |
| **DTEFTMVT** | Date du dernier mouvement réalisé sur le contrat |
| **DTOBTPDC** | Date obtention du permis de conduire |
| **DTPMRMCI** | Date de mise en circulation du véhicule |
| **DT_NAI** | Date de naissance |
| **IDECON** | Identifiant contrat |
| **IV** | Nombre de contrats individu-vie actifs |
| **MH** | Nombre de contrats multirisque habitation actifs |
| **MMJECHPP** | Date d’échéance du contrat (MMJJ) |
| **MTPAAREF** | Montant annuel de la prime de référence |
| **MTPAATTC** | Montant annuel de la prime |
| **NBCTACT** | Nombre de contrats actifs |
| **NBCTRES** | Nombre de contrats résiliés |
| **NIVBM** | Niveau du bonus malus |
| **NOTAREFF** | Numéro tarif, lié au bonus/malus, utilisé pour la tarification |
| **NO_AFR** | |
| **NUMFOY** | Numéro de foyer |
| **PUI_TRE** | Puissance fiscale du véhicule |
| **REGION** | Région |
| **RESAU4R** | Nombre de contrats auto résiliés |
| **RESDI** | Nombre de contrats divers résiliés |
| **RESIV** | Nombre de contrats individu-vie résiliés |
| **RESMH** | Nombre de contrats multirisque habitation résiliés |
| **RESSA** | Nombre de contrats santé résiliés |
| **RN_VL_VH** | Rang valeur du véhicule (cotation argus) |
| **SA** | Nombre de contrats actifs santé |
| **S_0_N** | Nb de sinistres dans les 12 derniers mois non responsables |
| **S_1_N** | Nb de sinistres dans les 12 mois de l’année dernière (glissante) non responsables |
| **S_2_N** | |
| **S_3_N** | |
| **S_0_O** | Nb de sinistres dans les 12 derniers mois responsables |
| **S_1_O** | |
| **S_2_O** | |
| **S_3_O** | |


## 🎯 Objectifs du projet
- 🔍 Analyser les données et identifier les variables les plus influentes
- 🔄 Appliquer des techniques de prétraitement des données
- 🤖 Entraîner et comparer plusieurs modèles de Machine Learning
- 📊 Évaluer les performances des modèles avec des métriques adaptées (AUC, précision, rappel, etc.)
- 📈 Interpréter les résultats pour une prise de décision optimisée

