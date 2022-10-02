# ML

## Data exploration

The customer-personality csv file contains information on customers such as ID, gender,
income, and number of purchases. From this, a company can understand customer behavior
for instance, to target specific type of customers for each of the product it sells. With the
data available, we can identify patterns among the customers.

We started by exploring the dataset to get a better understanding of it. As there is no y label,
meaning that there is no supervisor, we knew that we were dealing with unsupervised
learning. At this point we decided to perform clustering, which is the most common type of
unsupervised machine learning analysis to create categories grouping similar individuals. We
chose to go ahead with K-means as the clustering algorithm which will be discussed in more
detail.

After reading the data file into a pandas Dataframe, we explored the quality of the data and
the distribution of the variables. As the K-means algorithm is not able to deal with missing
values, and the “income” column contained NA’s, we decided to drop these, as seen in the
python notebook. Moreover, as categorical variables cannot be handled directly, we have
created dummy variables for these. We have also looked at the correlation matrix, which
shows moderate to high correlation between the variables.


Moving on, we looked at the distribution of all the variables. As the clusters have a spherical
shape, and for this to hold true, all variables should be normally distributed. As seen below,
our variables are normally distributed.

Lastly, we decided to scale the dataset so it would allow us to perform the next step, namely
PCA. The reason behind why we did this, is firstly that since PCA aims to find maximum
variance direction, then a variable with very high variance (ex : Mnt Wines) would account
for the first principal component, and it would not make sense to include other variables.
Moreover, the distance computation in k-Means assigns equal weight to each dimension.
Therefore, after scaling we get a better balance between variables.

## Feature extraction techniques

As our dataset has many features, it can become very difficult to cluster the observations. If
the distances between the observations are equal, they will all appear equally alike which
will not be meaningful for our analysis. To overcome the problems associated with the curse
of dimensionality, we will use dimensionality reduction techniques. These fall into two
categories- “Feature selection” and “Feature extraction”. For this analysis, we have chosen
to focus on the latter.

## Factor Analysis

Factor analysis on the other hand creates factors from the observed variables to represent
the common variance. The intuition here is that an “n” dimensional data can be represented
by “m” factors, where m < n.


We started by determining the number of factors that are useful in explaining the common
variance. We plotted the factors along with their eigenvalues as seen below. We selected
the number of factors with eigenvalues greater than 1 as we have standard scaled our
dataset. So, by selecting eigenvalues(variance) greater than 1, we are looking at factors that
explain more than a single observed variable.

From the graph, we can see that the eigenvalues drop below 1 after the 7th factor, we
therefore decide on 6 factors as the optimal number.

We can then look at the factor loadings which explain how much a factor explains a variable.
For example, the features “NumStorePurchases” and “Income” have high loadings which can
allow us to put these into the same factor, which can explain the common variance in people
who have high income and the number of purchases they make in store.

Moreover, this technique allows us to look at the variance explained by each factor. In total,
the 6 factors can explain approximately 40% of the total variance in our case. We can also
look at the values for the communalities, the proportion of each variable’s variance
explained by the factors. If we for instance consider “NumWebVisitsMonth”, around 79% of
its variance is explained by all the factors together.

With these machine learning techniques for unsupervised learning, we can get a better
sense of the dataset and better understand the customer behavior. This information can be
vital for the firm when deciding on how to target customers for their products and services.

## Principal Component Analysis (PCA)

We have used PCA to extract information about the hidden structure of the dataset. With
this, it allows us to see which dimensions best maximize the variance of the features by
projecting the data onto an M dimensional space.

After feature engineering and addition of dummy variables we have 36 variables, so if we
want to graph them all together, we will need the same number of dimensions. However,
with the PCA function from sklearn, we can transform the data.


Based on 50% explained variance rule of thumb we choose 5 components for this PCA
analysis.

## K-means Analysis

Following PCA, we use KElbowVisualizer to find the optimal number of clusters. This happens
to be 4 clusters.

We then perform k-means analysis to split the data into 4 clusters, using n_init=10,
max_iter=300.

To assess the results, we use Silhouette score, which indicates at how well the clusters are
separated from each other. Silhouette score here is 0.1 2 , which means a decent, although
far from perfect clustering. Silhouette score of 1 means clusters are perfectly separated,
while Silhouette score of -1 indicates there is no separation at all. A slightly positive


Silhouette score like we were able to achieve means there is some overlapping, but clusters
are indeed separated.

## Cluster Analysis

After some analysis of the clusters we were able to identify the following unique features:

- Cluster 1 : high income, no children, small household, single , highly spend on
    products , don’t look for deals , prefer to purchase at store
- Cluster 2 : less income , has at least one child , highest average family size , parent ,
    less spend on products , looks for deals , likely to purchase in store or web , frequent
- Cluster 3 : average income, likely to have at least one child , average family size ,
    single parent , average spend on products , highest total purchases , Phd
- Cluster 4 : less income , less spend on products , highest webvisits, attracted to deals,
    divorced, single parent , average family size

On the visualisations below it can be seen how each variable is distributed within every
cluster.
