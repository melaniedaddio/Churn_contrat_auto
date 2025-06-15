# Import des packages
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from itertools import combinations
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical
from skopt import BayesSearchCV
from sklearn.linear_model import LogisticRegression
from timeit import default_timer as timer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, auc, roc_curve
)
import plotly.graph_objects as go

def distrib_for_cat_by_target(var_cat: str, dataframe: pd.DataFrame, target: str):
    """
    Affiche la distribution normalisée d'une variable catégorielle par rapport à une variable cible binaire,
    ordonne les catégories selon leur fréquence globale, et affiche également la fréquence absolue pour chaque catégorie.
    
    La fonction calcule d'abord l'ordre des modalités de `var_cat` en se basant sur leur fréquence globale dans
    le DataFrame. Elle crée ensuite deux tableaux de contingence : l'un pour les fréquences absolues et l'autre pour
    les fréquences normalisées (pourcentages), en s'assurant que toutes les combinaisons de `target` et de `var_cat`
    apparaissent, même si certaines catégories ne sont présentes que pour un seul niveau de `target`.
    Les deux tableaux sont fusionnés et transformés en format long pour être visualisés avec Seaborn sous forme de graphique en
    barres. Chaque barre est annotée avec le pourcentage et la fréquence absolue (format : "xx.xx% (n)").
    
    Paramètres
    ----------
    var_cat : str
        Nom de la colonne de la variable catégorielle à analyser.
    dataframe : pd.DataFrame
        DataFrame contenant les données.
    target : str
        Nom de la colonne de la variable cible (binaire) utilisée pour le regroupement.
    
    Affichage
    ---------
    Affiche un graphique en barres illustrant la répartition en pourcentage et la fréquence absolue de chaque modalité de
    `var_cat` par rapport à `target`, avec les catégories ordonnées par fréquence décroissante.
    """
    # Ordre des catégories selon leur fréquence globale (du plus fréquent au moins fréquent)
    cat_order = dataframe[var_cat].value_counts().index.tolist()
    
    # Niveaux de target (on trie pour garantir l'ordre)
    target_levels = sorted(dataframe[target].unique())
    
    # Création des tableaux de contingence en valeur absolue et en pourcentage,
    # en s'assurant d'avoir toutes les combinaisons target x var_cat (avec des zéros si nécessaire)
    abs_table = pd.crosstab(dataframe[target], dataframe[var_cat]).reindex(index=target_levels, columns=cat_order, fill_value=0)
    norm_table = pd.crosstab(dataframe[target], dataframe[var_cat], normalize='index').reindex(index=target_levels, columns=cat_order, fill_value=0)
    
    # Transformation en format long pour fusionner les informations
    abs_table = abs_table.reset_index().melt(id_vars=target, var_name=var_cat, value_name='Absolute')
    norm_table = norm_table.reset_index().melt(id_vars=target, var_name=var_cat, value_name='Frequency')
    
    # Fusionner les deux tables sur target et var_cat
    merged_table = pd.merge(norm_table, abs_table, on=[target, var_cat])
    
    # S'assurer que l'ordre des catégories est respecté
    merged_table[var_cat] = pd.Categorical(merged_table[var_cat], categories=cat_order, ordered=True)
    merged_table = merged_table.sort_values(by=[target, var_cat])
    
    # Création du graphique avec Seaborn en précisant l'ordre des catégories pour l'argument hue
    g = sns.catplot(x=target, y="Frequency", hue=var_cat, data=merged_table, kind="bar",
                    height=8, aspect=2, hue_order=cat_order)
    ax = g.ax
    
    # Itérer sur les conteneurs de barres (un par niveau de var_cat, dans l'ordre de hue_order)
    # Chaque conteneur contient une barre par niveau de target (dans l'ordre des positions sur l'axe x)
    for hue_idx, container in enumerate(ax.containers):
        # Le niveau de var_cat correspondant (selon hue_order)
        current_cat = cat_order[hue_idx]
        for patch_idx, patch in enumerate(container):
            # On déduit la valeur de target à partir de l'ordre sur l'axe x
            # En supposant que les niveaux de target sont affichés dans l'ordre des ticks
            target_value = target_levels[patch_idx]
            # Rechercher dans merged_table la ligne correspondant à cette combinaison
            row = merged_table[(merged_table[target] == target_value) & (merged_table[var_cat] == current_cat)]
            if not row.empty:
                freq = row['Frequency'].values[0]
                absolute = row['Absolute'].values[0]
            else:
                freq = 0
                absolute = 0
            height = patch.get_height()
            annotation_text = f"{freq*100:.2f}% ({absolute})"
            ax.annotate(annotation_text,
                        (patch.get_x() + patch.get_width() / 2., height),
                        ha='center', va='center', fontsize=14, color='black',
                        xytext=(0, 20), textcoords='offset points')
    
    plt.title(f"Distribution de '{var_cat}' par 'Cible'", fontsize=22)
    plt.xlabel(target, fontsize=18)
    plt.ylabel('Fréquence', fontsize=18)
    plt.show()


def distrib_for_cont_by_target(cont_var: str, dataframe: pd.DataFrame, target: str, bins: int = 30)-> None:
    """
    Étudie la distribution d'une variable continue en fonction des modalités d'une variable binaire,
    en affichant les graphiques côte à côte dans une même figure.

    Pour chaque modalité de la variable binaire, la fonction crée un sous-graphique qui présente :
      - Un histogramme avec estimation de densité (KDE) de la variable continue,
      - Un rugplot pour visualiser la répartition brute des observations,
      - Des annotations affichant la moyenne et la médiane du groupe.

    Paramètres
    ----------
    cont_var : str
        Nom de la colonne de la variable continue à analyser.
    dataframe : pd.DataFrame
        DataFrame contenant les données.
    target : str
        Nom de la colonne de la variable binaire utilisée pour la segmentation.
    bins : int, optionnel
        Nombre de bins à utiliser pour l'histogramme (défaut : 30).

    Affichage
    ---------
    Affiche une figure avec un sous-graphique par modalité de la variable binaire, chacun illustrant
    la distribution (histogramme, KDE, rugplot) de la variable continue et les statistiques descriptives du groupe.
    """


    # Récupérer les modalités (niveaux) de la variable binaire et les trier
    target_levels = sorted(dataframe[target].unique())
    n_levels = len(target_levels)
    
    # Création d'une figure avec n sous-graphique(s) disposé(s) sur une seule ligne
    _, axes = plt.subplots(1, n_levels, figsize=(6 * n_levels, 6), sharey=True)
    
    # Si une seule modalité est présente, forcer axes à être une liste
    if n_levels == 1:
        axes = [axes]
    
    # Itérer sur chaque modalité et son axe associé
    for ax, level in zip(axes, target_levels):
        subset = dataframe[dataframe[target] == level]
        # Histogramme avec estimation de densité
        sns.histplot(data=subset, x=cont_var, bins=bins, stat="density", kde=True, ax=ax)
        # Ajout du rugplot pour visualiser la répartition brute
        sns.rugplot(data=subset, x=cont_var, ax=ax)
        
        # Calcul des statistiques descriptives
        mean_val = subset[cont_var].mean()
        median_val = subset[cont_var].median()
        annotation_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}"
        
        # Annotation dans le coin supérieur gauche du sous-graphique
        ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes,
                fontsize=12, verticalalignment="top", color="black")
        
        ax.set_title(f"{target} = {level}", fontsize=16)
        ax.set_xlabel(cont_var, fontsize=14)
        ax.set_ylabel("Densité", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def boxplot_by_target(var: str, dataframe: pd.DataFrame, target: str, quantiles: list = [1, 99]):
    """
    Affiche un boxplot de la variable explicative en fonction des modalités de la variable cible,
    en ajoutant des lignes horizontales pour la moyenne et pour les quantiles spécifiés.

    Paramètres
    ----------
    var : str
        Nom de la variable explicative à étudier.
    dataframe : pd.DataFrame
        DataFrame contenant les données.
    target : str
        Nom de la variable cible utilisée pour le regroupement.
    quantiles : list, optionnel
        Liste des quantiles (en pourcentage) à afficher. Par défaut, [1, 99].

    Affichage
    ---------
    Un boxplot avec la variable cible sur l'axe des x et la variable explicative sur l'axe des y,
    enrichi de repères indiquant la moyenne et les quantiles spécifiés.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Récupérer la colonne à analyser et calculer la moyenne
    col = dataframe[var]
    mean_val = col.mean()

    # Calculer les valeurs des quantiles spécifiés
    quantile_values = {q: np.percentile(col, q) for q in quantiles}

    # Création du boxplot
    plt.figure(figsize=(9, 7))
    ax = sns.boxplot(x=target, y=var, data=dataframe)

    # Affichage de la ligne de la moyenne
    ax.axhline(mean_val, ls='--', color='red', label=f"Mean = {mean_val:.2f}")

    # Définir une palette de couleurs pour les quantiles en fonction du nombre de quantiles
    colors = sns.color_palette("viridis", len(quantiles))
    
    # Afficher les lignes correspondant aux quantiles spécifiés
    for i, (q, value) in enumerate(quantile_values.items()):
        ax.axhline(value, ls='--', color=colors[i], label=f"P{q} = {value:.2f}")

    # Ajouter la légende et le titre
    ax.legend(loc='best')
    ax.set_title(f"Étude de la variable {var}\npar modalité de la variable '{target}'")
    plt.show()

def compute_df(df: pd.DataFrame, col: str, target_col: str) -> pd.DataFrame:
    """
    Calcule, pour chaque modalité/valeur de 'col', le nombre d'observations et le pourcentage de 1
    dans la variable cible 'target_col'.
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col : str
        Nom de la colonne explicative à étudier.
    target_col : str
        Nom de la variable cible binaire (ex. défaut = 1).
    
    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les colonnes :
          - col : modalité ou intervalle
          - counts_group : nombre de 1 dans target_col pour cette modalité
          - counts_total : nombre total d'observations dans cette modalité
          - pct : pourcentage de 1 dans cette modalité
    """
    # Nombre d'observations avec target==1 par modalité
    df_group = df[df[target_col] == 1].groupby(col, observed=False).size().reset_index(name='counts_group')
    # Nombre total d'observations par modalité
    df_total = df.groupby(col, observed=False).size().reset_index(name='counts_total')
    # Fusionner les deux résultats et calculer le pourcentage
    df_res = pd.merge(df_group, df_total, on=col, how='inner')
    df_res['pct'] = (df_res['counts_group'] / df_res['counts_total']) * 100
    df_res['pct'] = df_res['pct'].fillna(0)
    return df_res

def replace_neg(interval_str: str) -> str:
    """
    Remplace la borne inférieure négative d'un intervalle par 0.
    
    Paramètres
    ----------
    interval_str : str
        Représentation textuelle d'un intervalle (ex. "(-10, 20]").
    
    Retourne
    --------
    str
        Intervalle modifié si la borne inférieure est négative, sinon inchangé.
    """
    parts = interval_str.split(",")
    lower = float(parts[0].strip("(["))
    if lower < 0:
        return f"(0, {parts[1].strip()}"
    else:
        return interval_str

def compute_df_binned(df: pd.DataFrame, col: str, bins: int, target_col: str, cut: bool = True, q: int = 10) -> pd.DataFrame:
    """
    Effectue le binning d'une variable continue et calcule les taux de défaut pour chaque bin.
    Les valeurs spéciales (0 et -1) sont traitées séparément.
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col : str
        Nom de la variable continue à binner.
    bins : int
        Nombre de bins à utiliser pour les valeurs > 0.
    target_col : str
        Nom de la variable cible binaire.
    cut : bool, optionnel
        Si True, utilise pd.cut ; sinon, utilise pd.qcut. Par défaut True.
    q : int, optionnel
        Nombre de quantiles à utiliser avec qcut. Par défaut 10.
    
    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les résultats du calcul (effectifs et pourcentages) par bin.
    """
    frames = []
    # Binning pour les valeurs strictement positives
    df_pos = df[df[col] > 0].copy()
    if not df_pos.empty:
        df_pos['bin'] = pd.cut(df_pos[col], bins=bins, precision=0) if cut else pd.qcut(df_pos[col], q=q, precision=0, duplicates='drop')
        # Conversion en chaîne et remplacement de la borne inférieure négative si nécessaire
        df_pos['bin_str'] = df_pos['bin'].astype(str).apply(replace_neg)
        frames.append(compute_df(df_pos, 'bin_str', target_col))
    # Traitement des valeurs spéciales : -1 et 0
    for special in [-1, 0]:
        df_special = df[df[col] == special].copy()
        if not df_special.empty:
            df_special['bin_str'] = str(special)
            frames.append(compute_df(df_special, 'bin_str', target_col))
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()

def compute_df_categorical(df: pd.DataFrame, col: str, target_col: str) -> pd.DataFrame:
    """
    Calcule les taux de défaut pour une variable catégorielle.
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col : str
        Nom de la variable catégorielle.
    target_col : str
        Nom de la variable cible binaire.
    
    Retourne
    --------
    pd.DataFrame
        DataFrame avec effectifs et taux de défaut par modalité.
    """
    return compute_df(df, col, target_col)

# def plot_bin(df_res: pd.DataFrame, title: str, x_label: str, variable: str, pct_col: str = 'pct',
#              rotation: int = 0, xticks_font_size: int = 18, annot_font_size: int = 14,
#              unit: str = '%', loc: str = 'upper right'):
#     """
#     Représente graphiquement les résultats en affichant :
#       - Un diagramme en barres pour le nombre total d'observations par modalité.
#       - Un graphique linéaire (en superposition) du taux de défaut.
    
#     Paramètres
#     ----------
#     df_res : pd.DataFrame
#         DataFrame contenant les colonnes 'counts_total', 'counts_group' et le taux de défaut (pct).
#     title : str
#         Titre du graphique.
#     x_label : str
#         Label de l'axe des x.
#     variable : str
#         Nom de la colonne utilisée pour l'axe des x (ex. 'bin_str' ou la variable catégorielle).
#     pct_col : str, optionnel
#         Nom de la colonne contenant le taux de défaut (par défaut 'pct').
#     rotation : int, optionnel
#         Angle de rotation des labels de l'axe des x.
#     xticks_font_size : int, optionnel
#         Taille de la police des labels de l'axe des x.
#     annot_font_size : int, optionnel
#         Taille de la police pour les annotations sur le graphique.
#     unit : str, optionnel
#         Unité à afficher pour le taux (par défaut '%').
#     loc : str, optionnel
#         Position de la légende.
#     """
#     fontsize = 18
#     _, ax1 = plt.subplots(figsize=(20, 10))
#     ind = np.arange(len(df_res))
    
#     # Diagramme en barres pour le nombre total d'observations
#     ax1.bar(ind, df_res['counts_total'], color='tab:orange', width=0.9)
#     ax1.set_xlabel(x_label, fontsize=fontsize)
#     ax1.set_ylabel("Nb Total d'Observations par Tranche", color='tab:orange', fontsize=fontsize)
#     ax1.tick_params(axis='y', labelcolor='tab:orange', labelsize=fontsize)
#     ax1.set_xticks(ind)
#     ax1.set_xticklabels(df_res[variable], rotation=rotation, fontsize=xticks_font_size)
    
#     # Graphique linéaire pour le taux de défaut
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("% de défaut, par tranche", color='tab:blue', fontsize=fontsize)
#     p_line, = ax2.plot(ind, df_res[pct_col], color='tab:blue', marker='o')
#     ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=fontsize)
#     ax2.set_ylim(bottom=0)
    
#     # Annotation de chaque point du graphique linéaire
#     for i, pct in zip(ind, df_res[pct_col]):
#         ax2.annotate(f"{round(pct, 2)}{unit}", xy=(i, pct), xytext=(i, pct + 0.1), fontsize=annot_font_size)
    
#     # Ligne horizontale représentant le taux global de défaut
#     overall_pct = (df_res['counts_group'].sum() / df_res['counts_total'].sum()) * 100
#     p_hline = ax2.plot(ind, [overall_pct] * len(df_res), color='tab:red')
    
#     ax2.legend((p_line, p_hline[0]), ('Taux de défaut', 'Taux de défaut moyen'), loc=loc)
#     plt.title(title, fontsize=fontsize)
#     plt.show()

def plot_bin(df_res: pd.DataFrame, title: str, x_label: str, variable: str, pct_col: str = 'pct',
             rotation: int = 0, xticks_font_size: int = 18, annot_font_size: int = 14,
             unit: str = '%', loc: str = 'upper right'):
    """
    Représente graphiquement les résultats en affichant :
      - Un diagramme en barres pour le nombre total d'observations par modalité.
      - Un graphique linéaire (en superposition) du taux de défaut.
    """

    fontsize = 18

    # ⚠️ Vérification des données
    if df_res.empty:
        print("❌ df_res est vide — impossible de tracer le graphique.")
        return

    if 'counts_total' not in df_res or 'counts_group' not in df_res or pct_col not in df_res:
        print("❌ Colonnes manquantes dans df_res — vérifiez la structure du DataFrame.")
        return

    _, ax1 = plt.subplots(figsize=(20, 10))
    ind = np.arange(len(df_res))

    # Diagramme en barres
    ax1.bar(ind, df_res['counts_total'], color='tab:orange', width=0.9)
    ax1.set_xlabel(x_label, fontsize=fontsize)
    ax1.set_ylabel("Nb Total d'Observations par Tranche", color='tab:orange', fontsize=fontsize)
    ax1.tick_params(axis='y', labelcolor='tab:orange', labelsize=fontsize)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(df_res[variable], rotation=rotation, fontsize=xticks_font_size)

    # Courbe de taux de défaut
    ax2 = ax1.twinx()
    ax2.set_ylabel("% de défaut, par tranche", color='tab:blue', fontsize=fontsize)

    try:
        p_line, = ax2.plot(ind, df_res[pct_col], color='tab:blue', marker='o')
    except Exception as e:
        print(f"⚠️ Erreur lors du tracé de la courbe des taux de défaut : {e}")
        return

    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=fontsize)
    ax2.set_ylim(bottom=0)

    # Annotations de chaque point
    for i, pct in zip(ind, df_res[pct_col]):
        if pd.notna(pct):
            ax2.annotate(f"{round(pct, 2)}{unit}", xy=(i, pct), xytext=(i, pct + 0.1), fontsize=annot_font_size)

    # Ligne du taux global
    total = df_res['counts_total'].sum()
    group = df_res['counts_group'].sum()

    if total == 0 or pd.isna(total):
        overall_pct = 0
    else:
        overall_pct = (group / total) * 100

    p_hline = ax2.plot(ind, [overall_pct] * len(df_res), color='tab:red')

    ax2.legend((p_line, p_hline[0]), ('Taux de défaut', 'Taux de défaut moyen'), loc=loc)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_generic(dataframe: pd.DataFrame, var: str, target: str, bins: int = 10, force_binning: bool = True):
    """
    Analyse l'impact d'une variable explicative sur le taux de défaut.
    Si la variable possède plus de 100 modalités distinctes ou si force_binning est True, elle est traitée comme continue
    avec binning. Sinon, elle est traitée comme catégorielle.
    
    Paramètres
    ----------
    dataframe : pd.DataFrame
        DataFrame contenant les données.
    var : str
        Nom de la variable explicative.
    target : str
        Nom de la variable cible binaire (ex. défaut = 1).
    bins : int, optionnel
        Nombre de bins à utiliser pour une variable continue (par défaut 10).
    force_binning : bool, optionnel
        Si True, la variable sera traitée comme continue avec binning, même si elle a moins de 100 modalités distinctes.
    """
    print(f"Analyse des taux de défaut en fonction de la variable {var}")
    
    if force_binning or dataframe[var].nunique() > 100:
        try:
            df_res = compute_df_binned(dataframe, var, bins, target, cut=False, q=bins)
            x_var = 'bin_str'
        except:
            df_res = compute_df_categorical(dataframe, var, target)
            x_var = var  
    else:
        df_res = compute_df_categorical(dataframe, var, target)
        x_var = var
        
    title = "Taux de défaut"
    plot_bin(df_res, title, var, x_var, rotation=90)

def graph_correlations(list_cont_var: list, dataframe: pd.DataFrame, method: str = "pearson") -> None:
    """
    Affiche une heatmap des corrélations entre variables continues dans une figure compacte,
    en ajustant dynamiquement l'échelle de la police et la taille de la figure en fonction du nombre de variables.

    La fonction calcule la matrice de corrélation pour les variables spécifiées dans `list_cont_var`
    en utilisant la méthode indiquée (par défaut 'pearson'). L'échelle de la police et la taille de la figure
    sont ajustées de sorte que le texte reste lisible et que la figure offre suffisamment d'espace,
    même lorsque le nombre de variables varie.

    Parameters
    ----------
    list_cont_var : list
        Liste des noms de colonnes correspondant aux variables continues à analyser.
    dataframe : pd.DataFrame
        DataFrame contenant les données.
    method : str, optionnel
        Méthode de calcul de corrélation (exemple : 'pearson', 'spearman', 'kendall'). Par défaut 'pearson'.

    Returns
    --------
    None
        La fonction affiche la heatmap et ne retourne rien.
    """

    # Calculer la matrice de corrélation pour les variables continues spécifiées
    corrmat = dataframe[list_cont_var].corr(method=method)
    
    # Ajuster la taille de la figure et de la police selon le nombre de variables à étudier
    n = len(list_cont_var)
    figsize = (max(10, n * 1.2), max(10, n * 1.2))
    plt.figure(figsize=figsize)
    font_scale = max(0.8, 10 / n)
    sns.set_theme(font_scale=font_scale, style="whitegrid")
    
    # Afficher la heatmap avec annotations et une palette de couleurs "coolwarm"
    sns.heatmap(corrmat, annot=True, cmap="coolwarm", square=True)
    
    # Afficher le graphique
    plt.show()

def cramers(var1: str, var2: str, dataframe: pd.DataFrame):
    """
    Calcule le coefficient de Cramer pour deux variables catégorielles.

    Parameters:
        var1 (str): Nom de la première variable catégorielle.
        var2 (str): Nom de la deuxième variable catégorielle.
        dataframe (pd.DataFrame): DataFrame contenant les variables.

    Returns:
        tuple: (var1, var2, abs_V_cramer, chi2, prob_chi2)
               - abs_V_cramer : coefficient de Cramer arrondi à 3 décimales.
               - chi2        : statistique du test du chi2 arrondie à 3 décimales.
               - prob_chi2   : p-value du test arrondie à 4 décimales ou '< 0.0001' si très faible.
    """
    # Création de la table de contingence entre les deux variables
    crosstab = pd.crosstab(dataframe[var1], dataframe[var2], rownames=[var1], colnames=[var2])
    
    # Calcul du test du chi2 sur la table de contingence
    chi2, p_value, _, _ = ss.chi2_contingency(crosstab)
    
    # Nombre total d'observations
    n = crosstab.values.sum()
    
    # Calcul du V de Cramer
    # min(crosstab.shape)-1 correspond au degré de liberté minimum
    min_dim = min(crosstab.shape) - 1
    v_cramer = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
    abs_v_cramer = round(abs(v_cramer), 3)
    
    # Formatage de la p-value
    prob_chi2 = round(p_value, 4)
    if prob_chi2 < 0.0001:
        prob_chi2 = '< 0.0001'
        
    return var1, var2, abs_v_cramer, round(chi2, 3), prob_chi2

def cramers_v_between_all(list_var_category: list, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le V de Cramer pour chaque paire unique de variables catégorielles explicatives.

    Parameters:
        list_var_category (list): Liste des noms des variables catégorielles.
        dataframe (pd.DataFrame): DataFrame contenant les variables.

    Returns:
        pd.DataFrame: DataFrame avec les résultats pour chaque paire, comportant les colonnes :
                      ['Variable_1', 'Variable_2', 'abs_V_cramer', 'Chi2', 'Prob_Chi2']
    """
    results = []
    # Utilisation de combinations pour générer toutes les paires uniques de variables
    for var1, var2 in combinations(list_var_category, 2):
        results.append(cramers(var1, var2, dataframe))
    
    # Création du DataFrame des résultats
    df_results = pd.DataFrame(results, columns=['Variable_1', 'Variable_2', 'abs_V_cramer', 'Chi2', 'Prob_Chi2'])
    # Suppression des doublons (au cas où) et tri décroissant par V de Cramer
    df_results = df_results.drop_duplicates(subset=['abs_V_cramer', 'Chi2', 'Prob_Chi2']).sort_values('abs_V_cramer', ascending=False)
    
    return df_results

def cramers_v_with_target(list_var_category: list, target: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le V de Cramer entre chaque variable catégorielle de la liste et une variable cible.

    Parameters:
        list_var_category (list): Liste des noms des variables catégorielles.
        target (str): Nom de la variable cible.
        dataframe (pd.DataFrame): DataFrame contenant les variables.

    Returns:
        pd.DataFrame: DataFrame avec les résultats comportant les colonnes :
                      ['Variable', 'abs_V_cramer', 'Chi2', 'Prob Chi2']
    """
    results = []
    for var in list_var_category:
        results.append(cramers(var, target, dataframe))
    
    # Création du DataFrame avec une colonne superflue "Variable 2" qui sera ensuite retirée
    df_results = pd.DataFrame(results, columns=['Variable', 'Variable_2', 'abs_V_cramer', 'Chi2', 'Prob_Chi2'])
    # Retrait de la colonne redondante et tri décroissant par V de Cramer
    df_results = df_results.drop(columns='Variable_2').sort_values('abs_V_cramer', ascending=False)
    
    return df_results

def rename_field_categories(dataframe:pd.DataFrame, field:str, init_values: list, new_values: list, replace:bool =True)-> pd.DataFrame:
    """
    Labellise les valeurs d'une colonne numérique en les découpant en intervalles et en attribuant des labels.

    Cette fonction utilise pd.cut pour segmenter la colonne spécifiée (field) en intervalles définis par init_values,
    puis affecte à chaque intervalle le label correspondant dans new_values. Si replace est True, la colonne d'origine
    est remplacée par les catégories labellisées ; sinon, une nouvelle colonne nommée 'LABEL_<field>' est créée.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Le DataFrame contenant les données.
    field : str
        Le nom de la colonne à labelliser.
    init_values : list
        Les bornes pour découper la colonne (doit contenir n+1 valeurs pour n intervalles).
    new_values : list
        Les labels à attribuer aux intervalles (doit contenir n valeurs).
    replace : bool, optionnel
        Si True, remplace la colonne d'origine par la version labellisée (défaut : True).

    Returns
    --------
    pandas.DataFrame
        Le DataFrame mis à jour avec la colonne labellisée ou avec une nouvelle colonne 'LABEL_<champ>'.
    """
    categories = pd.cut(dataframe[field], init_values, labels=new_values)
    if replace:
        dataframe[field] = categories
    else:
        dataframe['LABEL_' + field] = categories
    return dataframe

def extract_corr_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation d'un DataFrame et la reformate en un DataFrame plus lisible.

    La fonction réalise les opérations suivantes :
      - Calcule la matrice de corrélation à l'aide de df.corr().
      - Masque le triangle inférieur et la diagonale de la matrice (les remplaçant par NaN) afin de ne conserver
        que la partie supérieure.
      - Convertit la matrice masquée en un DataFrame à trois colonnes : 
          - 'var_1' : première variable
          - 'var_2' : seconde variable
          - 'corr'  : coefficient de corrélation entre les deux variables.
      - Ajoute une colonne 'corr_abs' contenant la valeur absolue de la corrélation.
      - Trie le DataFrame par 'corr_abs' en ordre décroissant, puis par 'var_1' en ordre croissant.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les variables pour lesquelles la corrélation doit être calculée.

    Returns
    --------
    pandas.DataFrame
        Un DataFrame contenant trois colonnes ('var_1', 'var_2', 'corr') ainsi qu'une colonne 'corr_abs' qui
        représente la valeur absolue de la corrélation, trié par ordre décroissant de 'corr_abs'.
    """
    # Calcul de la matrice de corrélation
    cor = df.corr()
    
    # Création d'un masque pour le triangle inférieur et la diagonale (valeurs à masquer)
    mask = np.tril(np.ones(cor.shape, dtype=bool))
    
    # Masquer le triangle inférieur et la diagonale de la matrice
    corr = cor.mask(mask)
    
    # Convertir la matrice masquée en DataFrame avec les paires de variables et leur corrélation
    corr_df = (
        corr.stack()                                  # Empile la matrice pour obtenir une Series avec MultiIndex
            .sort_values(ascending=False)             # Trie par ordre décroissant de corrélation
            .reset_index()                             # Réinitialise l'index pour obtenir un DataFrame
            .rename(columns={'level_0': 'var_1', 'level_1': 'var_2', 0: 'corr'})
    )
    
    # Ajouter une colonne avec la valeur absolue de la corrélation
    corr_df['corr_abs'] = corr_df['corr'].abs()
    
    # Trier le DataFrame par valeur absolue de corrélation (décroissant) puis par var_1 (croissant)
    corr_df.sort_values(['corr_abs', 'var_1'], ascending=[False, True], inplace=True)
    
    return corr_df

def create_dummies(dataframe: pd.DataFrame, list_var_cat: list, drop_original_var_cat: bool = True) -> pd.DataFrame:
    """
    Crée des variables dummies pour les colonnes catégorielles d'un DataFrame.

    Cette fonction génère des variables binaires (dummies) à partir des colonnes spécifiées,
    ajoute ces variables au DataFrame original et, optionnellement, supprime les colonnes
    originales.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Le DataFrame contenant les données.
    list_var_cat : list
        La liste des noms de colonnes catégorielles à transformer en variables dummies.
    drop_original_var_cat : bool, optional
        Indique si les colonnes originales doivent être supprimées après la création des dummies.
        Par défaut, True.

    Returns
    -------
    pd.DataFrame
        Le DataFrame mis à jour avec les variables dummies ajoutées et, si spécifié,
        les colonnes originales supprimées.
    """
    # Créer les variables dummies avec un préfixe personnalisé pour chaque colonne
    dummies = pd.get_dummies(dataframe[list_var_cat],
                             prefix=["DUMMY_" + col for col in list_var_cat])
    # Nettoyer les noms de colonnes pour les rendre compatibles
    dummies.columns = dummies.columns.str.replace(r"[^\w]", "_", regex=True)
    # Ajouter les dummies au DataFrame original
    dataframe = dataframe.join(dummies)
    # Supprimer les colonnes originales si demandé
    if drop_original_var_cat:
        dataframe = dataframe.drop(columns=list_var_cat)
    return dataframe

def graph_conf_mat(conf_mat: confusion_matrix):
    """
    Affiche une matrice de confusion au format graphique

    Parameters
    ----------
    conf_mat : array-like of shape (2, 2)
        Matrice de confusion issue de sklearn.metrics.confusion_matrix.

    Affiche
    -------
    Un graphique matplotlib.
    """
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 12},
        xticklabels=["Prévu: Non défaut", "Prévu: Défaut"],
        yticklabels=["Réel: Non défaut", "Réel: Défaut"]
    )

    plt.title("Matrice de confusion", fontsize=14, weight='bold')
    plt.xlabel("Prédiction", fontsize=12)
    plt.ylabel("Réalité", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()

def custom_classification_report(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    Calcule et affiche des métriques de classification (précision, rappel, f1-score)
    pour chaque classe

    Parameters
    ----------
    y_true : variable cible (observée)
    y_pred : prédiction de la variable cible

    Returns
    --------
    pd.DataFrame
        Un tableau contenant les scores de précision, rappel et f1-score
        pour chaque classe : "Non défaut (classe 0)" et "Défaut (classe 1)".
    """
    metrics = {
        "Non défaut (classe 0)": {
            "precision": precision_score(y_true, y_pred, pos_label=0),
            "recall": recall_score(y_true, y_pred, pos_label=0),
            "f1-score": f1_score(y_true, y_pred, pos_label=0),
        },
        "Défaut (classe 1)": {
            "precision": precision_score(y_true, y_pred, pos_label=1),
            "recall": recall_score(y_true, y_pred, pos_label=1),
            "f1-score": f1_score(y_true, y_pred, pos_label=1),
        }
    }

    df = pd.DataFrame(metrics).T.round(3)
    return df

def plot_roc_curve_interactive(y_true: pd.Series, y_prob: pd.Series):
    """
    Affiche une courbe ROC interactive avec Plotly, avec hover, AUC et ligne de base.

    Parameters
    ----------
    y_true : pd.Series
        Vraies étiquettes binaires (0 ou 1).
    y_prob : pd.Series
        Probabilités prédites pour la classe positive.

    Returns
    -------
    None
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # Courbe ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f"Courbe ROC (AUC = {roc_auc:.3f})",
        line=dict(color="darkblue", width=2),
        hovertemplate="FPR = %{x:.3f}<br>TPR = %{y:.3f}<extra></extra>"
    ))

    # Diagonale "aléatoire"
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))

    fig.update_layout(
        title="📈 Courbe ROC interactive",
        xaxis_title="Taux de faux positifs (FPR)",
        yaxis_title="Taux de vrais positifs (TPR)",
        xaxis=dict(range=[0, 1], tickformat=".1f"),
        yaxis=dict(range=[0, 1], tickformat=".1f"),
        template="plotly_white",
        width=700,
        height=500
    )

    fig.show()

def find_best_threshold_f1(y_true: pd.Series, y_prob: pd.Series, display_graph: bool = True) -> tuple[float, float]:
    """
    Affiche l'évolution du F1-score en fonction du seuil de score
    et retourne le seuil optimal avec le F1-score maximum.

    Parameters
    ----------
    y_true : array-like
        Vraies classes (0 ou 1).
    y_prob : array-like
        Probabilités du modèle pour la classe positive (1).
    display_graph : bool
        Affiche ou non le graphique F1-score vs seuil.

    Returns
    -------
    f1_max : float
        Meilleure valeur de F1-score atteinte.
    opti_threshold : float
        Seuil qui maximise le F1-score.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))

    idx_best = np.argmax(f1_scores)
    opti_threshold = thresholds[idx_best]
    f1_max = f1_scores[idx_best]

    if display_graph:
        plt.figure(figsize=(10, 7))
        plt.plot(thresholds, f1_scores, label="F1-score", color="darkblue", linewidth=2)
        plt.scatter(opti_threshold, f1_max, color='red', s=40, zorder=5)

        # Traits pointillés
        plt.axhline(f1_max, linestyle='--', color='red', linewidth=1)
        plt.axvline(opti_threshold, linestyle='--', color='red', linewidth=1)

        # Texte positionné dans les marges (hors du plot)
        plt.annotate(f"{opti_threshold:.3f}",
                     xy=(opti_threshold, 0), xycoords='data',
                     xytext=(0, -10), textcoords='offset points',
                     ha='center', va='top', fontsize=9, color='red',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.annotate(f"{f1_max:.3f}",
                     xy=(0, f1_max), xycoords='data',
                     xytext=(-10, 0), textcoords='offset points',
                     ha='left', va='center', fontsize=9, color='red',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.title("F1-score en fonction du seuil", fontsize=12)
        plt.xlabel("Seuil de score", fontsize=11)
        plt.ylabel("F1-score", fontsize=11)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(bottom=0, top=1.02)
        plt.xlim(left=0, right=1)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return f1_max, opti_threshold

def compute_metric_by_threshold(
    y_true: pd.Series,
    y_prob: pd.Series,
    metric: str = "f1"
) -> pd.DataFrame:
    """
    Calcule une métrique de classification (précision, rappel ou F1-score)
    en fonction d'une série de seuils.

    Parameters
    ----------
    y_true : pd.Series
        Vraies étiquettes binaires (0 ou 1).
    y_prob : pd.Series
        Probabilités prédites pour la classe positive.
    metric : str
        Nom de la métrique à calculer : 'precision', 'recall' ou 'f1'.

    Returns
    -------
    pd.DataFrame
        Un DataFrame avec deux colonnes : threshold et valeur de la métrique.
    """
    thresholds = np.linspace(0.01, 0.99, 100)

    # Choix de la fonction associée à la métrique
    if metric == "precision":
        scoring_func = precision_score
    elif metric == "recall":
        scoring_func = recall_score
    elif metric == "f1":
        scoring_func = f1_score
    else:
        raise ValueError("La métrique doit être 'precision', 'recall' ou 'f1'.")

    scores = [
        scoring_func(y_true, (y_prob >= s).astype(int), pos_label=1, zero_division=0)
        for s in thresholds
    ]

    return pd.DataFrame({
        "threshold": thresholds,
        metric: scores
    })


def plot_all_metrics_interactive(
    y_true: pd.Series,
    y_prob: pd.Series,
    list_metrics: list = ["precision", "recall", "f1"]
):
    """
    Affiche un graphique interactif Plotly avec les courbes de précision, rappel et F1-score.

    Parameters
    ----------
    y_true : pd.Series
        Vraies étiquettes binaires (0 ou 1).
    y_prob : pd.Series
        Probabilités pour la classe positive.
    list_metrics : list of str
        Liste des métriques à afficher : 'precision', 'recall', 'f1'.

    Returns
    -------
    None
    """
    fig = go.Figure()

    for metric in list_metrics:
        df = compute_metric_by_threshold(y_true, y_prob, metric)

        fig.add_trace(go.Scatter(
            x=df["threshold"],
            y=df[metric],
            mode="lines+markers",
            name=metric.capitalize(),
            hovertemplate=metric.capitalize() + " = %{y:.3f}<extra></extra>"
        ))

    fig.update_layout(
        title="📈 Évolution des métriques selon le seuil",
        xaxis_title="Seuil de score",
        yaxis_title="Valeur de la métrique",
        xaxis=dict(tickformat=".2f"),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(title="Métrique"),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        width=850
    )

    fig.show()

    
def compute_all_metrics_by_threshold(
    y_true: pd.Series,
    y_prob: pd.Series
) -> pd.DataFrame:
    """
    Calcule précision, rappel et F1-score pour chaque seuil entre 0.01 et 0.99
    et les fusionne dans un seul DataFrame.

    Parameters
    ----------
    y_true : pd.Series
        Vraies classes binaires (0 ou 1).
    y_prob : pd.Series
        Probabilités prédites pour la classe positive.

    Returns
    -------
    pd.DataFrame
        Un DataFrame avec les colonnes :
        'threshold', 'precision', 'recall', 'f1'
    """
    df_precision = compute_metric_by_threshold(y_true, y_prob, metric="precision")
    df_recall= compute_metric_by_threshold(y_true, y_prob, metric="recall")
    df_f1= compute_metric_by_threshold(y_true, y_prob, metric="f1")

    df_metrics = df_precision.copy()
    df_metrics["recall"] = df_recall["recall"]
    df_metrics["f1"] = df_f1["f1"]

    return df_metrics