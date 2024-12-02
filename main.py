#%%
# =========================================================================
# Import des librairies
# =========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# =========================================================================
# Import des données data dictionary
# =========================================================================
df_structure = pd.read_csv("Data_dictionary.csv")
pd.set_option("display.max_colwidth", None)

# %%
# =========================================================================
# Import des données country data
# =========================================================================
df_country = pd.read_csv("Country_data.csv")

# %%
# =========================================================================
# Visualisation des données
# =========================================================================

# Drop non-numeric columns for correlation heatmap
numeric_df = df_country.select_dtypes(include=[np.number])

plt.figure(figsize=(15,10))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


# Create scatter plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_country, x="gdpp", y="child_mort")
plt.title('Child Mortality vs GDP per capita')
plt.xlabel('GDP per capita')
plt.ylabel('Child Mortality Rate')
plt.show()

# %%
# =========================================================================
# Préparation des données
# =========================================================================
df_country["exports"] = df_country["exports"] * df_country["gdpp"]
df_country["imports"] = df_country["imports"] * df_country["gdpp"]
df_country["health"] = df_country["health"] * df_country["gdpp"]
df_country.head()

# %%
# =========================================================================
# Normalisation
# =========================================================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df_country.iloc[:,1:-1])

# %%
# =========================================================================
# réduction dimensionnelle
# =========================================================================
from sklearn.decomposition import PCA

pca = PCA(random_state = 42)
pca.fit(X)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Nombre de dimensions après réduction")
plt.ylabel("Variance cumulée")
plt.show()

# %%
# =========================================================================
# 3 dimentions
# =========================================================================
pca = PCA(n_components=3)
X = pca.fit_transform(X)

df_pca = pd.DataFrame(X, columns=["PC1", "PC2", "PC3"])
df = pd.concat([df_pca], axis=1)  # Remove the country column since it's non-numeric
df.head()

plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.show()
# %%
# =========================================================================
# Modélisation
# =========================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Création du scatter plot
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                    c=df_country['gdpp'],  # Colorer selon le PIB
                    cmap='viridis')

# Ajout des labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Visualisation 3D des pays selon les composantes principales')

# Ajouter une barre de couleur
plt.colorbar(scatter, label='PIB par habitant')

plt.show()

# %%
# =========================================================================
# Détection d'anomalies
# =========================================================================
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
# étape 1 -> instanciation
lof = LocalOutlierFactor(n_neighbors = 10)
isol_forest = IsolationForest(contamination = 0.02, random_state = 42)

# étape 2 -> entraînement
lof.fit(X)
isol_forest.fit(X)

# étape 3 -> détection des anomalies
scores = lof.negative_outlier_factor_
outliers_lof = np.argwhere( scores > np.percentile(scores, 98))

outliers = isol_forest.predict(X)
outliers_forest = np.argwhere(outliers == -1)

# étape 4 -> suppression des anomalies
X1 = np.delete(X, outliers_lof, axis = 0)
X2 = np.delete(X, outliers_forest, axis = 0)

# %%
# =========================================================================
# Visualisation des anomalies
# =========================================================================

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection = "3d")
ax.scatter(X1[:,0], X1[:,1], X1[:,2], color = "red", alpha = 0.5, label = "LOF")
ax.scatter(X2[:,0], X2[:,1], X2[:,2], color = "green", alpha = 1, label = "Isolation Forest", s = 150, marker="*")
plt.legend()
plt.show()
# %%
# =========================================================================
# Clustering
# =========================================================================
X = np.copy(X2)

from sklearn.cluster import MeanShift

ms = MeanShift()
ms.fit(X)
predict = ms.predict(X)
centers = ms.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], X[:,2], c = predict)
ax.scatter(centers[:,0], centers[:,1], centers[:,2], marker = "x", color = "red", s = 150)

# %%
# =========================================================================
# Méthode du coude
# =========================================================================

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Méthode du coude pour trouver un nombre de clusters optimal
inertias = []

for num_clusters in list(range(1,9)):
    kmeans = KMeans(n_clusters = num_clusters, init = "k-means++", random_state = 42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    inertias.append(kmeans.inertia_)

    if (num_clusters > 1):
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"Pour {num_clusters} clusters nous avons coeff de silhouette = {np.round(silhouette_avg, decimals = 3)*100}%")

plt.figure()
plt.plot(range(1,9), inertias)
plt.show()

# %%
# %%
# =========================================================================
# Clustering avec 6 clusters
# =========================================================================

kmeans = KMeans(n_clusters = 6, init = "k-means++", random_state = 42)
kmeans.fit(X)
cluster_predict = kmeans.predict(X)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], X[:,2], c = cluster_predict, s = 50, alpha = 1)
# %%
# =========================================================================
# conversion en dataframe
# =========================================================================
df_pca = pd.DataFrame(X, columns = ["PC1", "PC2", "PC3"])
df_predict = pd.DataFrame(cluster_predict, columns = ["cluster"])
df = pd.concat([df_country.iloc[:,0], df_pca, df_predict], axis = 1)
df.head(10)
# %%
# =========================================================================
# Interprétation des clusters
# =========================================================================
df_country = pd.merge(df_country, df.drop(["PC1", "PC2", "PC3"], axis = 1), on = "country")
df_country.dropna(inplace = True)
df_country.head()


# %%
# =========================================================================
# Agrégation des données par cluster
# =========================================================================
df_country["cluster"] = df_country["cluster"].astype(int)
clustered_data = df_country.select_dtypes(include=['int64', 'float64']).groupby("cluster").mean()

# %%
# =========================================================================
# Visualisation des agrégats
# =========================================================================

cluster_id = range(0,6)

plt.figure(figsize = (8,8))
plt.subplot(2,2,1)
plt.bar(cluster_id, clustered_data["child_mort"], color = "red")
plt.xlabel("Cluster ID")
plt.ylabel("Mortalité infantile moyenne")
plt.xticks(cluster_id)

plt.subplot(2,2,2)
plt.bar(cluster_id, clustered_data["life_expec"], color = "purple")
plt.xlabel("Cluster ID")
plt.ylabel("Espérance de vie moyenne")
plt.xticks(cluster_id)

plt.subplot(2,2,3)
plt.bar(cluster_id, clustered_data["income"], color = "maroon")
plt.xlabel("Cluster ID")
plt.ylabel("Revenu moyen")
plt.xticks(cluster_id)

plt.subplot(2,2,4)
plt.bar(cluster_id, clustered_data["total_fer"], color = "orange")
plt.xlabel("Cluster ID")
plt.ylabel("Nb enfants moyen")
plt.xticks(cluster_id)

plt.show()

# %%
# =========================================================================
# Pays appartenant au cluster 1
# =========================================================================

prior_countries = np.array(df_country)
prior_countries = prior_countries[prior_countries[:,-1]==1][:,0]

print(prior_countries)

# %%
# =========================================================================
# Pays appartenant au cluster 2
# =========================================================================

prior_countries = np.array(df_country)
prior_countries = prior_countries[prior_countries[:,-1]==2][:,0]

print(prior_countries)

# %%
# =========================================================================
# Analyse des associations
# =========================================================================

from mlxtend.frequent_patterns import apriori, association_rules
df = df_country.iloc[:,1:-1]
df.head(10)

# %%
# =========================================================================
# Conversion des variables continues en variables catégoriques
# =========================================================================

cat_df = pd.DataFrame()

for col in range(len(df.iloc[0,:])):
    min_val = df.iloc[:,col].min(axis=0)
    max_val = df.iloc[:,col].max(axis=0)
    mean_val = df.iloc[:,col].mean(axis=0)
    if (col == 0 or col == 5 or col == 7):
        cat_df[f"{df.iloc[:,col].name}"] = pd.cut(df.iloc[:,col],
                                                 bins = [min_val, mean_val, max_val],
                                                 labels = [False, True])
    else:
        cat_df[f"{df.iloc[:,col].name}"] = pd.cut(df.iloc[:,col],
                                                 bins = [min_val, mean_val, max_val],
                                                 labels = [True, False])
cat_df.head(10)

# %%
# =========================================================================
# Suppression des valeurs manquantes
# =========================================================================

cat_df.dropna(inplace = True)
cat_df.isna().sum()
cat_df.head(10)

# %%
# =========================================================================
# Extraction des itemsets fréquents
# =========================================================================

frequent_itemsets = apriori(cat_df, min_support = 0.5, use_colnames = True)
frequent_itemsets.head(20)

# %%
# =========================================================================
# Extraction des règles d'association
# =========================================================================

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=frequent_itemsets)
rules.sort_values("confidence", ascending=False)

# %%
# =========================================================================
# Visualisation et interprétation des règles d'association
# =========================================================================

# Afficher les règles les plus pertinentes
print("\nTop 10 règles avec la plus grande confiance :")
print(rules.nlargest(10, 'confidence')[['antecedents', 'consequents', 'confidence', 'lift']])

# Visualisation des règles avec un scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence pour les règles d\'association')

# Ajouter une ligne pour le seuil de confiance
plt.axhline(y=0.7, color='r', linestyle='--', label='Seuil de confiance (0.7)')
plt.legend()
plt.show()

# Visualisation des règles les plus fortes
plt.figure(figsize=(12, 8))
rules_viz = rules.nlargest(10, 'lift')
sns.barplot(x=range(len(rules_viz)), y=rules_viz['lift'])
plt.xticks(rotation=45)
plt.title('Top 10 règles selon le lift')
plt.xlabel('Règle')
plt.ylabel('Lift')
plt.tight_layout()
plt.show()

# %%
