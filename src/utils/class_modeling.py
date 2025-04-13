from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from time import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report, 
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix,
    precision_score, 
    recall_score, 
)
import plotly.graph_objects as go
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparator:
    """
    Classe pour comparer plusieurs modèles de classification avec optimisation bayésienne,
    visualiser leurs performances (ROC, PR) et identifier le meilleur seuil F1-score.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        random_state: int = 0,
        list_models: Optional[List[str]] = None
    ):
        """
        Initialise la classe avec les jeux de données.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series
        X_test  : pd.DataFrame
        y_test  : pd.Series
        random_state : int
            Graine de reproductibilité
        list_models : list, optional
            Liste des modèles à tester (ex : ["Random Forest", "Logistic Regression"]).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.scoring = "f1"
        self.list_models = list_models
        self.opt_by_model: Dict[str, BayesSearchCV] = {}
        self.proba_dict: Dict[str, np.ndarray] = {}
        self.df_resultats: pd.DataFrame = pd.DataFrame()

        # Obtenir tous les modèles disponibles
        self.all_model_spaces = self._get_model_spaces()

        # Expansion des modèles "généraux"
        if list_models:
            expanded = []
            for model in list_models:
                if model == "Logistic Regression":
                    expanded.extend([
                        "Logistic Regression (L1)",
                        "Logistic Regression (L2)",
                        "Logistic Regression (ElasticNet)"
                    ])
                else:
                    expanded.append(model)

            # Vérification validité
            unknown = [m for m in expanded if m not in self.all_model_spaces]
            if unknown:
                raise ValueError(f"Modèle(s) inconnu(s) : {unknown}\nModèles disponibles : {list(self.all_model_spaces.keys())}")
            self.selected_models = {k: v for k, v in self.all_model_spaces.items() if k in expanded}
        else:
            self.selected_models = self.all_model_spaces

    def _get_model_spaces(self) -> Dict[str, tuple]:
        """
        Définit les modèles et leur espace d'hyperparamètres.
        """
        common_tree_params = {
            'n_estimators': Integer(100, 300),
            'max_depth': Integer(3, 15),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2']),
        }

        return {
            "Random Forest": (
                RandomForestClassifier(random_state=self.random_state, verbose=0),
                {**common_tree_params, 'bootstrap': Categorical([True])}
            ),
            "Extra Trees": (
                ExtraTreesClassifier(random_state=self.random_state, verbose=0),
                {**common_tree_params, 'bootstrap': Categorical([True])}
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=self.random_state, verbose=0),
                {**common_tree_params, 'learning_rate': Real(0.01, 0.3, prior='log-uniform')}
            ),
            "LightGBM": (
                LGBMClassifier(random_state=self.random_state, verbose=0),
                {
                    'n_estimators': Integer(100, 300),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'max_depth': Integer(3, 15),
                    'num_leaves': Integer(15, 100),
                    'min_child_samples': Integer(5, 20)
                }
            ),
            "Logistic Regression (L1)": (
                LogisticRegression(max_iter=1000, random_state=self.random_state),
                {
                    'penalty': Categorical(['l1']),
                    'solver': Categorical(['liblinear', 'saga']),
                    'C': Real(0.001, 10, prior='log-uniform'),
                    'tol': Real(1e-4, 1e-2, prior='log-uniform')
                }
            ),
            "Logistic Regression (L2)": (
                LogisticRegression(max_iter=1000, random_state=self.random_state),
                {
                    'penalty': Categorical(['l2']),
                    'solver': Categorical(['newton-cg', 'lbfgs', 'sag', 'saga']),
                    'C': Real(0.001, 10, prior='log-uniform'),
                    'tol': Real(1e-4, 1e-2, prior='log-uniform')
                }
            ),
            "Logistic Regression (ElasticNet)": (
                LogisticRegression(max_iter=1000, random_state=self.random_state),
                {
                    'penalty': Categorical(['elasticnet']),
                    'solver': Categorical(['saga']),
                    'C': Real(0.001, 10, prior='log-uniform'),
                    'l1_ratio': Real(0.1, 0.9, prior='uniform'),
                    'tol': Real(1e-4, 1e-2, prior='log-uniform')
                }
            )
        }

    def get_proba_dict(self) -> Dict[str, np.ndarray]:
        """
        Calcule les probabilités prédites sur X_test pour chaque modèle.
        """
        self.proba_dict = {
            name: opt.best_estimator_.predict_proba(self.X_test)[:, 1]
            for name, opt in self.opt_by_model.items()
        }
        return self.proba_dict
    

    def bayes_optimize_models(
        self,
        n_iter: int = 50,
        cv_folds: int = 5,
        scoring: str = "f1"
    ) -> pd.DataFrame:
        """
        Effectue l’optimisation bayésienne sur une liste de modèles.

        Returns
        -------
        pd.DataFrame :
            Résumé des performances (score CV + durée).
        """
        self.scoring = scoring
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        models = self._get_model_spaces()

        if self.list_models:
            expanded = []
            for model in self.list_models:
                if model == "Logistic Regression":
                    expanded += [
                        "Logistic Regression (L1)",
                        "Logistic Regression (L2)",
                        "Logistic Regression (ElasticNet)"
                    ]
                else:
                    expanded.append(model)
            models = {k: v for k, v in models.items() if k in expanded}

        results = []
        for name, (estimator, space) in models.items():
            print(f"🔍 Optimisation en cours : {name}")
            start = time()
            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=space,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=0,
                random_state=self.random_state
            )
            opt.fit(self.X_train, self.y_train)
            duration = round(time() - start, 2)

            results.append({
                "Modèle": name,
                f"Score {scoring.upper()} (CV)": round(opt.best_score_, 8),
                "Durée (s)": duration
            })
            self.opt_by_model[name] = opt

        self.df_resultats = pd.DataFrame(results).sort_values(by=f"Score {scoring.upper()} (CV)", ascending=False)
        self.proba_dict = self.get_proba_dict()
        return self.df_resultats

    def get_best_model(self) -> Dict[str, Union[str, float, object]]:
        """
        Retourne le meilleur modèle selon la métrique d'évaluation (scoring).

        Returns
        -------
        dict :
            {
                "model_name" : str → nom du meilleur modèle,
                "score"      : float → score CV associé,
                "estimator"  : object → objet sklearn entraîné (best_estimator_)
            }
        """
        if self.df_resultats.empty:
            raise ValueError("Aucun résultat disponible. Veuillez exécuter bayes_optimize_models() d'abord.")

        # Nom de la colonne de score (ex: "Score F1 (CV)")
        score_col = [col for col in self.df_resultats.columns if "Score" in col][0]
        best_row = self.df_resultats.loc[self.df_resultats[score_col].idxmax()]

        model_name = best_row["Modèle"]
        best_score = best_row[score_col]
        best_estimator = self.opt_by_model[model_name].best_estimator_

        print(f"Meilleur modèle : {model_name}")
        print(f"Score ({self.scoring.upper()} CV) : {best_score:.4f}")

        return {
            "model_name": model_name,
            "score": best_score,
            "estimator": best_estimator,
            "y_proba": self.proba_dict[model_name]
        }

    def plot_roc_curve_interactive(self) -> None:
        """
        Affiche les courbes ROC triées par AUC.
        """
        fig = go.Figure()
        data = []

        for model_name, y_prob in self.proba_dict.items():
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc_score = auc(fpr, tpr)
            data.append((model_name, fpr, tpr, auc_score))

        for model_name, fpr, tpr, auc_score in sorted(data, key=lambda x: x[3], reverse=True):
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{model_name} (AUC = {auc_score:.3f})",
                hovertemplate="FPR = %{x:.2f}<br>TPR = %{y:.2f}<extra></extra>"
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name="Modèle aléatoire",
            line=dict(dash='dash', color='gray'),
            hoverinfo='skip'
        ))

        fig.update_layout(
            title="📊 Courbes ROC comparées",
            xaxis_title="Taux de faux positifs (FPR)",
            yaxis_title="Taux de vrais positifs (TPR)",
            template="plotly_white",
            width=850,
            height=520
        )
        fig.show()

    def plot_all_bayes_convergences(self) -> None:
        """
        Affiche un graphique interactif Plotly avec les courbes de convergence
        de tous les modèles optimisés avec BayesSearchCV.

        Cela permet de visualiser l’évolution du score (ex : F1, AUC...) à chaque itération
        pour comparer la stabilité et la vitesse de convergence de chaque modèle.
        """
        if not self.opt_by_model:
            print("❌ Aucun modèle n'a été optimisé. Lancez compare_models() d'abord.")
            return

        fig = go.Figure()

        for model_name, opt in self.opt_by_model.items():
            scores = opt.cv_results_["mean_test_score"]
            iterations = list(range(1, len(scores) + 1))

            best_iter = int(np.argmax(scores)) + 1
            best_score = np.max(scores)

            # Trace principale
            fig.add_trace(go.Scatter(
                x=iterations,
                y=scores,
                mode='lines+markers',
                name=f"{model_name} (max={best_score:.4f})",
                hovertemplate="Itération = %{x}<br>Score = %{y:.4f}<extra></extra>"
            ))

            # Marqueur du meilleur score
            fig.add_trace(go.Scatter(
                x=[best_iter],
                y=[best_score],
                mode="markers",
                name=f"{model_name} ★",
                marker=dict(color='red', size=8, symbol='star'),
                showlegend=False,
                hovertemplate=f"{model_name}<br>Itération optimale = {best_iter}<br>Score = {best_score:.4f}<extra></extra>"
            ))

        # Label de score
        scoring = list(self.opt_by_model.values())[0].scoring
        score_label = {
            'roc_auc': 'ROC AUC',
            'f1': 'F1-score',
            'accuracy': 'Accuracy',
            'precision': 'Précision',
            'recall': 'Recall'
        }.get(scoring, scoring.upper())

        fig.update_layout(
            title=f"📉 Courbes de convergence Bayésienne ({score_label})",
            xaxis_title="Itérations",
            yaxis_title=score_label,
            template="plotly_white",
            width=950,
            height=550,
            legend_title="Modèles"
        )

        fig.show()


    def plot_precision_recall_curve_interactive(self) -> None:
        """
        Affiche les courbes Précision-Rappel triées par Average Precision.
        """
        fig = go.Figure()
        data = []

        for model_name, y_prob in self.proba_dict.items():
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            ap = average_precision_score(self.y_test, y_prob)
            data.append((model_name, precision, recall, ap))

        for model_name, precision, recall, ap in sorted(data, key=lambda x: x[3], reverse=True):
            fig.add_trace(go.Scatter(
                x=recall, y=precision, mode='lines',
                name=f"{model_name} (AP = {ap:.3f})",
                hovertemplate="Recall = %{x:.2f}<br>Précision = %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title="📈 Courbes Précision-Rappel comparées",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_white",
            width=850,
            height=520
        )
        fig.show()

    def best_f1_by_model(self) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Calcule le seuil optimal qui maximise le F1-score pour chaque modèle.

        Returns
        -------
        dict :
            {
                "f1_scores": {modèle: f1_max},
                "opti_thresholds": {modèle: seuil_optimal}
            }
        """
        seuils = {}
        f1_scores = {}
        curves_data = []

        for model_name, y_prob in self.proba_dict.items():
            thresholds = np.linspace(0.01, 0.99, 100)
            f1_list = [f1_score(self.y_test, (y_prob >= t).astype(int)) for t in thresholds]
            idx_best = int(np.argmax(f1_list))
            seuils[model_name] = thresholds[idx_best]
            f1_scores[model_name] = f1_list[idx_best]

            curves_data.append({
                "model": model_name,
                "thresholds": thresholds,
                "f1_scores": f1_list,
                "f1_max": f1_list[idx_best]
            })

        fig = go.Figure()
        for data in sorted(curves_data, key=lambda x: x["f1_max"], reverse=True):
            fig.add_trace(go.Scatter(
                x=data["thresholds"],
                y=data["f1_scores"],
                mode="lines+markers",
                name=f"{data['model']} (max={data['f1_max']:.3f})",
                hovertemplate="Seuil = %{x:.2f}<br>F1-score = %{y:.3f}<extra></extra>"
            ))

        fig.update_layout(
            title="🎯 F1-score en fonction du seuil (trié par performance)",
            xaxis_title="Seuil de décision",
            yaxis_title="F1-score",
            template="plotly_white",
            width=950,
            height=520,
            yaxis=dict(range=[0, 1.02])
        )
        fig.show()

        return {
            "f1_scores": f1_scores,
            "opti_thresholds": seuils
        }

    def evaluate_model(self, model_name: str, threshold: Optional[float] = None) -> None:
        """
        Affiche un rapport HTML interprétable pour le modèle sélectionné.

        Parameters
        ----------
        model_name : str
            Le nom du modèle tel qu'il apparaît dans self.proba_dict.
        threshold : float, optional
            Seuil à utiliser pour binariser les probabilités (défaut = 0.5).
        """
        if model_name not in self.proba_dict:
            print(f"❌ Modèle {model_name} introuvable.")
            return

        y_prob = self.proba_dict[model_name]
        y_pred = (y_prob >= (threshold if threshold else 0.5)).astype(int)

        # Métriques
        acc = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_prob)
        report = classification_report(self.y_test, y_pred, output_dict=True, target_names=["Non défaut", "Défaut"])

        # Construction tableau HTML
        html = f"""
        <div class="alert alert-success" style="font-family:Arial;">
        <h4>📊 Évaluation du modèle : <code>{model_name}</code></h4>

        <h5>⚙️ Métriques globales</h5>
        <table border="1" style="border-collapse:collapse; width:100%; text-align:left;">
            <thead style="background-color:#f2f2f2;">
            <tr><th style="padding:6px;">Métrique</th><th style="padding:6px;">Valeur</th></tr>
            </thead>
            <tbody>
            <tr>
                <td style="padding:6px;">Accuracy</td>
                <td style="padding:6px;">{acc:.3f}</td>

            </tr>
            <tr>
                <td style="padding:6px;">ROC AUC</td>
                <td style="padding:6px;">{auc_score:.3f}</td>
            </tr>
            </tbody>
        </table>

        <h5 style="margin-top:20px;">🎯 Métriques par classe</h5>
        <table border="1" style="border-collapse:collapse; width:100%; text-align:left;">
            <thead style="background-color:#f2f2f2;">
            <tr>
                <th style="padding:6px;">Classe</th>
                <th style="padding:6px;">Précision</th>
                <th style="padding:6px;">Rappel</th>
                <th style="padding:6px;">F1-score</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td style="padding:6px;"><b>Non défaut</b></td>
                <td style="padding:6px;">{report['Non défaut']['precision']:.3f}</td>
                <td style="padding:6px;">{report['Non défaut']['recall']:.3f}</td>
                <td style="padding:6px;">{report['Non défaut']['f1-score']:.3f}</td>
            </tr>
            <tr>
                <td style="padding:6px;"><b>Défaut</b></td>
                <td style="padding:6px;">{report['Défaut']['precision']:.3f}</td>
                <td style="padding:6px;">{report['Défaut']['recall']:.3f}</td>
                <td style="padding:6px;">{report['Défaut']['f1-score']:.3f}</td>
            </tr>
            </tbody>
        </table>
        </div>
        """

        display(HTML(html))


    def graph_conf_mat(self, model_name: str, threshold: float = 0.5) -> None:
        """
        Affiche la matrice de confusion du modèle spécifié à un seuil donné.

        Parameters
        ----------
        model_name : str
            Le nom du modèle tel qu’il figure dans self.proba_dict.
        threshold : float, default=0.5
            Seuil à appliquer aux probabilités pour obtenir les prédictions binaires.
        """
        if model_name not in self.proba_dict:
            print(f"❌ Modèle '{model_name}' non trouvé dans les probabilités.")
            return

        y_prob = self.proba_dict[model_name]
        y_pred = (y_prob >= threshold).astype(int)
        conf_mat = confusion_matrix(self.y_test, y_pred)

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

        plt.title(f"Matrice de confusion - {model_name}", fontsize=14, weight='bold')
        plt.xlabel("Prédiction", fontsize=12)
        plt.ylabel("Réalité", fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.show()


    def get_metrics_by_threshold(
        self,
        model_name: str,
        list_metrics: List[str] = ["precision", "recall", "f1"]
    ) -> Optional[pd.DataFrame]:
        """
        Calcule les valeurs des métriques choisies (precision, recall, f1)
        pour une série de seuils appliqués à un modèle donné.

        Parameters
        ----------
        model_name : str
            Nom du modèle à analyser (doit exister dans self.proba_dict).
        list_metrics : list of str
            Métriques à inclure dans le tableau : 'precision', 'recall', 'f1'.

        Returns
        -------
        pd.DataFrame or None :
            DataFrame avec les colonnes : 'threshold', 'precision', 'recall', 'f1'.
        """
        if model_name not in self.proba_dict:
            print(f"❌ Modèle '{model_name}' non trouvé dans les probabilités.")
            return None

        y_prob = self.proba_dict[model_name]
        thresholds = np.linspace(0.01, 0.99, 100)
        data = {"threshold": thresholds}

        for metric in list_metrics:
            if metric == "precision":
                data["precision"] = [precision_score(self.y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
            elif metric == "recall":
                data["recall"] = [recall_score(self.y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
            elif metric == "f1":
                data["f1"] = [f1_score(self.y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
            else:
                print(f"⚠️ Métrique non reconnue : {metric}. Ignorée.")

        return pd.DataFrame(data)


    def plot_metrics_by_threshold(
        self,
        model_name: str,
        list_metrics: List[str] = ["precision", "recall", "f1"]
    ) -> None:
        """
        Affiche un graphique interactif Plotly des courbes précision / rappel / F1-score
        en fonction du seuil de décision pour un modèle donné.

        Parameters
        ----------
        model_name : str
            Nom du modèle à analyser (doit exister dans self.proba_dict).
        list_metrics : list of str
            Métriques à afficher : parmi 'precision', 'recall', 'f1'.
        """
        # 🔍 Récupérer les métriques
        df = self.get_metrics_by_threshold(model_name, list_metrics=list_metrics)
        if df is None:
            return

        fig = go.Figure()

        for metric in list_metrics:
            if metric not in df.columns:
                continue

            fig.add_trace(go.Scatter(
                x=df["threshold"],
                y=df[metric],
                mode="lines+markers",
                name=metric.capitalize(),
                hovertemplate=f"Seuil = %{{x:.2f}}<br>{metric.capitalize()} = %{{y:.3f}}<extra></extra>"
            ))

        fig.update_layout(
            title=f"📈 {model_name} — Évolution des métriques selon le seuil",
            xaxis_title="Seuil de score",
            yaxis_title="Valeur de la métrique",
            xaxis=dict(tickformat=".2f"),
            yaxis=dict(range=[0, 1.05]),
            legend=dict(title="Métriques"),
            template="plotly_white",
            hovermode="x unified",
            height=500,
            width=850
        )

        fig.show()


    def generate_model_report(
        self,
        n_iter: int = 50,
        cv_folds: int = 5,
        scoring: str = "f1",
    ) -> None:
        """
        Génère un rapport complet automatisé

        Parameters
        ----------
        n_iter : int
            Nombre d'itérations pour l'optimisation Bayesienne.
        cv_folds : int
            Nombre de folds pour la validation croisée.
        scoring : str
            Métrique utilisée pour la comparaison (par défaut "f1").
        """
        print("🔧 Étape 1 : Optimisation Bayésienne des modèles...\n")
        df_scores = self.bayes_optimize_models(n_iter=n_iter, cv_folds=cv_folds, scoring=scoring)
        display(df_scores)

        print("\n📊 Étape 2 : Prédiction des probabilités...")
        self.get_proba_dict()

        print("\n📉 Étape 3 : Affichage des courbes ROC & Précision-Rappel")
        self.plot_roc_curve_interactive()
        self.plot_precision_recall_curve_interactive()

        print("\n🎯 Étape 4 : Optimisation du F1-score par modèle")
        f1_results = self.best_f1_by_model()

        # Meilleur modèle selon F1
        best_model = max(f1_results["f1_scores"], key=f1_results["f1_scores"].get)
        best_f1 = f1_results["f1_scores"][best_model]
        best_thresh = f1_results["opti_thresholds"][best_model]

        print(f"\n✅ Étape 5 : Évaluation du meilleur modèle : {best_model}")
        print(f"   → F1-score max = {best_f1:.4f} atteint pour un seuil = {best_thresh:.3f}\n")

        self.evaluate_model(model_name=best_model, threshold=best_thresh)
        self.graph_conf_mat(model_name=best_model, threshold=best_thresh)
        self.plot_metrics_by_threshold(model_name=best_model)
