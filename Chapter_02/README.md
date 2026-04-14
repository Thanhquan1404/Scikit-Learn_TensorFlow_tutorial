# End-to-End Machine Learning Project

> 
> 
> - ***General step to be a data scientist in a real estate company:***
>     - ***Look at the big picture.***
>     - ***Get the data.***
>     - ***Discover and visualize the data to gain insights.***
>     - ***Prepare the data for Machine Learning algorithm.***
>     - ***Select a model and train it.***
>     - ***Fine-tune your model.***
>     - ***Present your solution.***
>     - ***Launch, monitor, and maintain your system.***

## Working with Real Data

> 
> 
> - ***There are thousands of open datasets to choose from, ranging across all sorts of domains. Here are a few places you can look to get data:***
>     - ***Popular open data repositories:***
>         - ***UC Irvine Machine Learning Repository***
>         - ***Kaggle datasets***
>         - ***Amazon’s AWS datasets***
>     - ***Meta portals (they list open data repositories):***
>         - ***http://dataportal.org/***
>         - ***http://opendatamonitor.eu/***
>         - ***http://quandl.com/***
>     - ***Other page listing many popular open data repositories:***
>         - ***Wikipedia’s list of Machine Learning datasets***
>         - [***Quora.com](http://Quora.com) question***
>         - ***Dataset subreddit***

    In this chapter we chose the California Housing Prices dataset from the StatLib repository2 (see Figure 2-1). This dataset was based on data from the 1990 California census. It is not exactly recent (you could still afford a nice house in the Bay Area at the time), but it has many qualities for learning, so we will pretend it is recent data. We also added a categorical attribute and removed a few features for teaching purposes.

## Look at the Big Picture

- The first task you are asked to perform is to ***build a model of housing prices in California using the California census data***. This data has metrics such as the population, median income, median housing price, and so on for each block group in California. Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). We will just call them “districts” for short.
- Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.

## Frame the Problem

- The first question to ask your boss is ***what exactly is the business objective***; building a model is probably not the end goal. ***How does the company expect to use and benefit from this model?*** This is important because it will ***determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it.***

> 
> 
> - ***Pipeline:***
>     - ***sequence of data processing components is called a data pipeline. Pipelines are very common in Machine Learning systems, since there is a lot of data to manipulate and many data transformations to apply.***
>     - ***Each component pulls in a large amount of data (asynchronous), processes it, and spits out the result in another data store, and then some time later the next component in the pipeline pulls this data and spits out its own output, and so on.***
>     - ***the interface between components is simply the data store.***

> 
> 
> - Questions when we want to start design our system will be made:
>     - is it supervised, unsupervised, or reinforcement learning?
>     - Is it a classification task, a regression task, or something else?
>     - Should you use batch learning or online learning techniques?

## Select Performance Measure

- Because of median price prediction, RMSE or Root Mean Square Error could be a best choice to model tuning action via decrease as much as RMSE of model.
- Your next step is to select a performance measure. A typical performance measure for regression problems is the Root Mean Square Error (RMSE). It gives an idea of how much error the system typically makes in its predictions, with a higher weight for large errors.

$$\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}}$$

- We also using MAE if you realize that your data supposes to contain many outlier districts.
$$ \text{MAE}(y, \hat{y}) = \frac{\sum_{i=0}^{N - 1} |y_i - \hat{y}_i|}{N}$$

- Both the RMSE and the MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target values.
    - Computing the root of a sum of squares $\text{RMSE}$ corresponds to the Euclidian norm: it is the notion of distance you are familiar with. It is also called the $l2$ norm, noted ${\lVert . \lVert}_2$ (or just ${\lVert . \lVert}$).
    - Computing the sum of absolutes $\text{MAE}$ corresponds to the $l1$ norm, noted $\lVert . \lVert _1$. It is sometimes called the $Manhattan$ norm because it measures the distance between two points in a city if you can only travel along orthogonal city blocks.

    - More generally, the $l_k$ norm of vector $v$ containing $n$ elements is defined as:
$$\lVert v \rVert_k = (|v_0|^k+|v_1|^k+ ... + |v_n|^k)^{\frac {1} {k}}$$

## Check the Assumptions

- ***The district prices that your system outputs are going to be fed into a downstream Machine Learning system, and we assume that these prices are going to be used as such.***
- ***what if the downstream system actually converts the prices into categories (e.g., “cheap,” “medium,” or “expensive”) and then uses those categories instead of the prices them‐selves? ⇒ Your system just needs to get the category right. If that’s so, then the problem should have been framed as a classification task, not a regression task. You don’t want to find this out after working on a regression system for months.***

## Get the Data

### Create the Workspace

- You will need a number of Python modules: Jupyter, NumPy, Pandas, Matplotlib, and Scikit-Learn.
- You can use your system’s packaging system (e.g., apt-get on Ubuntu, or MacPorts or HomeBrew on macOS), install a Scientific Python distribution such as Anaconda and use its packaging system, or just use Python’s own packaging system, pip, which is included by default with the Python binary installers (since Python 2.7.9).6 You can check to see if **`*pip*`** is installed by typing the following command:

```bash
pip3 --version
pip3 install --upgrade pip
```

- Create Isolated Environment via Venv Python Virtual Machine
```bash
python3 -m venv venv
source venv/bin/activate
```
- Install and running Jupyter Notebook on localhost
```bash
pip install jupyter
pip install notebook
```
- Let follow these steps below to make a notebook Jupyter:
    - $\text{File} \to \text{New} \to {notebook} \to \text{double taps to open file}$
    - $\text{New} \to \text{Select kernel}$

### Download the Data
```python
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetching_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # tạo thư mục nếu chưa tồn tại
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    # đường dẫn file .tgz local
    tgz_path = os.path.join(housing_path, "housing.tgz")

    # tải file
    urllib.request.urlretrieve(housing_url, tgz_path)

    # giải nén
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

fetching_housing_data()
```

### Take a Quick Look at the Data Structure

```python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
```

- Each row represents one district. There are 10 attributes (you can see the first 6 in the screenshot): ***longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, and ocean_proximity. And The info() method is useful to get a quick description of the data***

```python
housing.info()
```

- There are ***20,640 instances in the dataset***, which means that it is fairly small by Machine Learning standards, but it’s perfect to get started. Notice that the total_bed rooms attribute has only ***20,433 non-null values***, meaning that ***207 districts are missing this feature***. We will need to take care of this later.
- You can find out what categories exist and how many districts belong to each category by using the value_counts() method.

```python
housing["ocean_proximity"].value_counts()
```

```python
#The count, mean, min, and max rows are self-explanatory.
housing.describe()
```

- A histogram shows the number of instances (on the vertical axis) that have a given value range (on the horizontal axis). You can either plot this one attribute at a time, or you can call the hist() method on the whole dataset, and it will plot a histogram for each numerical attribute.

```python
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20, 15))
plt.show()
```

- Well, this works, but it is not perfect: if you run the program again, it will generate a different test set! Over time, you (or your Machine Learning algorithms) will get to see the whole dataset, which is what you want to avoid.
- Using with id table to make stable of data split when we want to keep the same testing-training data structure even if there is any changes or not

```python
import hashlib

def test_set_check(identifier, test_ratio, hash):
  return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
  return data.loc[~in_test_set], data.loc[in_test_set]
```

```python
housing["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(data=housing, test_ratio=0.2, id_column="id")r
```

- Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways. The simplest function is train_test_split, which does pretty much the same thing as the function split_train_test defined earlier, with a couple of additional features.

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

- Suppose you chatted with experts who told you that the median income is a very important attribute to predict median housing prices. You may want to ensure that the test set is representative of the various categories of incomes in the whole dataset.
- This means that you should not have too many strata, and each stratum should be large enough. The following code creates an income category attribute by dividing the median income by 1.5 (to limit the number of income categories), and rounding up using ceil (to have discrete categories), and then merging all the categories greater than 5 into category 5:

```python
housing['income_cat'] = np.ceil( housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5.0, 5, inplace=True)
housing['income_cat'].hist(bins=5)
plt.show()
```
```python 
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(housing, housing['income_cat']):
  strat_train_set = housing.loc[train_idx]
  strat_test_set = housing.loc[test_idx]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```
```python
for set_ in (strat_train_set, strat_test_set):
  set_.drop('income_cat', axis=1, inplace=True)
```
## Discover and Visualize the Data to Gain Insights

- First, make sure you have put the test set aside and you are only exploring the training set. Also, if the training set is very large, you may want to sample an exploration set, to make manipulations easy and fast.

```python
housing = strat_train_set.copy()
```

### Visualizing Geographical Data

- Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data

```python
housing.plot(kind="scatter", x="longitude", y="latitude"
```

- Let set the $alpha$ to 0.1 to see the desity of graph

```python
housing.plot(
	kind="scatter", 
	x="longitude", 
	y="latitude", 
	alpha=0.1
)
```

- Now that’s much better: you can clearly see the high-density areas, namely the Bay Area and around Los Angeles and San Diego, plus a long line of fairly high density in the Central Valley, in particular around Sacramento and Fresno.
- The radius of each circle represents the ***district’s population (option s),*** and the color represents ***the price (option c).*** We will use a predefined ***color map (option cmap) called jet,*** which ranges from blue (low values) to red (high prices).

```python
housing.plot(
	kind='scatter',
	x='longitude',
	y='latitude',
	alpha=0.4,
	s=housing['population']/100, label="population", 
	figsize=(10,7),
	c="median_house_value",
	cmap=plt.get_cmap("jet"),
	colorbar=True,
)
plt.legend()
```

- This image tells you that the housing prices are very much related to the location (e.g., close to the ocean) and to the population density, as you probably knew already. It will ***probably be useful to use a clustering algorithm to detect the main clusters,*** and add new features that measure the proximity to the cluster centers.

### Looking for Correlations

- Since the dataset is not too large, you can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the corr() method:

```python
corr_matrix = housing.corr()
```

- The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value.

```python
numeric_data = housing.drop(columns=['ocean_proximate']
corr_matrix = numeric_data.corr()
corr_matrix.sort_values(ascending=False)
```

> The correlation coefficient only measures linear correlations (“if x
> goes up, then y generally goes up/down”). It may completely miss
> out on nonlinear relationships (e.g., “if x is close to zero then y generally goes up”).
>

```python
from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12,6))
```

```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)
```

## Prepare the Data for Machine Learning Algorithm

> 
> 
> - ***This will allow you to reproduce these transformations easily on any dataset (e.g., the next time you get a fresh dataset)***
> - ***You will gradually build a library of transformation functions that you can reuse in future projects.***
> - ***You can use these functions in your live system to transform the new data before feeding it to your algorithm.***
> - ***This will make it possible for you to easily try various transformations and see which combination of transformations works best.***

```python
housing = strat_train_set.drop('median_house_value', axis=1)
housing_label = strat_train_set['median_house_value'].copy()
```

### Data Cleaning

     The new bedrooms_per_room attribute is much more correlated with the median house value than the total number of rooms or bedrooms.

> 
> 
> - ***Get rid of the corresponding districts.***
> - ***Get rid of the while attribute***
> - ***Set the values to some value (zero, the mean, the median, etc)***

     If you choose option 3, you should compute the median value on the training set, and use it to fill the missing values in the training set, but also don’t forget to save the median value that you have computed. You will need it later to replace missing values in the test set when you want to evaluate your system, and also once the system goes live to replace missing values in new data.

```python
housing.dropna(subset=['total_bedrooms'])   # option 1
housing.drop('total_bedrooms', axis=1)      # option 2
median = housing['total_bedrooms'].median() # option 3
housing['total_bedrooms'].fillna(median, inplace=True)
```

     Scikit-Learn provides a handy class to take care of missing values: Imputer.

```python
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
```

```python
from sklearn.preprocessing import Imputer

housing_num = housing.drop('ocean_procimity', axis=1)
imputer = Imputer(strategy='median')
imputer.fit(housing_num)
```

The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable.

```python
X = imputer.transform(housing_num)
```

- ***Scikit-Learn Design***
  - Consistency: All object share a consistent and simple interface
      - **`*Estimators:*`** **`*fit()*`** method: that takes only one parameter  or two parameter is the training dataset and training label in supervised learning.
      - **`*Transformers:*`** Some estimators (such as an imputer) can also transform a dataset; these are called **`*transformers()*`**. This transformation generally relies on the learned parameters, as is the case for an imputer. All transformers also have a convenience method called **`*fit_transform()*`** that is equivalent to calling fit() and then transform() (but sometimes fit_transform() is optimized and runs much faster).
      - **`*Predictor:`*** Finally, some estimators are capable of making predictions given a dataset; they are called predictors. A predictor has a **`*predict()*`** method that takes a dataset of new instances and returns a dataset of corresponding predictions. It also has a **`*score()*`** method that measures the quality of the predictions given a test set (and the corresponding labels in the case of supervised learning algorithms).
  - **`*Inspection:*`** All the estimator’s hyperparameters are accessible directly via public instance variables (e.g., imputer.strategy), and all the estimator’s learned parameters are also accessible via public instance variables with an underscore suffix (e.g., imputer.statistics_).
  - **`*Nonproliferation of classes:*`** Datasets are represented as NumPy arrays or SciPy sparse matrices, instead of homemade classes. Hyperparameters are just regular Python strings or numbers.
  - **`*Composition:*`** Existing building blocks are reused as much as possible. For example, it is easy to create a Pipeline estimator from an arbitrary sequence of transformers followed by a final estimator, as we will see.
  - **`*Sensible defaults:*`** Scikit-Learn provides reasonable default values for most parameters, making it easy to create a baseline working system quickly.

### Handling Text and Categorical Attributes

```python
housing_cat = housing['ocean_proximity']
housing_cat.head(10)
```
    

As most machine learning, there are usually numeric datasets only, in this case we could use factorize() to analyze ‘ocean proximity’ categorical column.

```python
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]
```

ML algorithms will assume that two nearby values are more similar than two distant values. Obviously this is not the case (for example, categories 0 and 4 are more similar than categories 0 and 2). To fix this issue, a common solution is to create one binary attribute per category: one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is “NEAR OCEAN” (and 0 otherwise), and so on. This is called one-hot encoding, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold).

```python
from sklearn.preprocessing import OneHotEncoder 
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot
```
After one-hot encoding we get a matrix with thousands of columns, and the matrix is full of zeros except for a single 1 per row. Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the nonzero elements.

> NumPy’s reshape() function allows one dimension to be -1, which means “unspecified”: the value is inferred from the length of the array and the remaining dimensions.
>

```python
from sklearn.preprocessing import CategoricalEncoder 
cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_car_1hot
```

### Custom Transformers


> Although Scikit-Learn provides many useful transformers, you will need to write
> your own for tasks such as ***custom cleanup operations or combining specific
> attributes.***


> 
> 
> - Base class for all estimators in scikit-learn. Inheriting from this class provides default implementations of:
>     - setting and getting parameters used by `GridSearchCV` and friends;
>     - textual and HTML representation displayed in terminals and IDEs;
>     - estimator serialization;
>     - parameters validation;
>     - data validation;
>     - feature names validation.
>     - Have two extra methods ***(get_params() and set_params(**dict)***

```python
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
		self.add_bedrooms_per_room = add_bedrooms_per_room
	def fit(self, X, y=none):
		return self #nothing else to do
	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]
	
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

> Transformer has one hyper-parameter, add_bedrooms_per_room,
> set to True by default (it is often helpful to provide sensible defaults). This hyperpara‐
> meter will allow you to easily find out whether adding this attribute helps the
> Machine Learning algorithms or not.

### Feature Scaling

> One of the most important transformations you need to apply to your data is ***feature
> scaling.*** With few exceptions, Machine Learning algorithms don’t perform well when
> the input numerical attributes have very different scales. 

> There are two common ways to get all attributes to have the same scale: **`*min-max
> scaling*`** and **`*standardization*`**. 
- Min-max scaling (many people call this normalization) is quite simple: values are shifted and rescaled so that they end up ranging from 0 to 1. ***We do this by subtracting the min value and dividing by the max minus the min.*** Scikit-Learn provides a transformer called **`*MinMaxScaler*`** for this. It has a feature_range hyper-parameter that lets you change the range if you don’t want **`*0–1*`** for some reason.

$$
x_{scaled} = \frac {x_i - Min} {Max - Min}
$$

- **`*Standardization*`** is quite different: ***first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the variance so that the resulting distribution has unit variance.*** Unlike min-max scaling, ***standardization does not bound values to a specific range***, which may be a problem for some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1). However, standardization is much less affected by outliers.

$$
  z = \frac{x - \mu}{\sigma}
$$

### Transformation Pipelines

> As you can see, there are many data transformation steps that need to be executed in the right order. Fortunately, Snikit-Learn provides the Pipeline class to help with such sequences of transformation
> 

```python
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
	('imputer', Imputer(strategy="median")),
	('attribs_adder', CombinedAttributesAdder()),
	('std_scaler', StandardScaler()),	
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
```

- When you call the pipeline’s fit() method, it calls fit_transform() sequentially on all transformers, passing the output of each call as the parameter to the next call, until it reaches the final estimator, for which it just calls the fit() method.
- Now it would be nice if we could feed a Pandas DataFrame containing non-numerical columns directly into out pipeline, instead of having to first manually extract the numerical columns into a NumPy array.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self,X):
		return X[self.attribute_names].values
```

```python
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
	('selector', DataFrameSelector(num_attribs)),
	('imputer', Imputer(strategy="median")),
	('attribs_adder', CombinedAttributesAdder()),
	('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
	('selector', DataFrameSelector(cat_attribs)),
	('cat_encoder', CategoricalEncoder(encoding='onehot-dense')),
])
```

> But how can you join these two pipelines into a single pipeline? The answer is to use
> Scikit-Learn’s FeatureUnion class. You give it a list of transformers (which can be
> entire transformer pipelines); when its transform() method is called, it runs each
> transformer’s transform() method in parallel, waits for their output, and then con‐
> catenates them and returns the result (and of course calling its fit() method calls
> each transformer’s fit() method).
```python
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transform_list=[
	("num_pipeline", num_pipeline),
	("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
```

## Select and Train a Model

### Training and Evaluating on the Training Set

- Let’s first train a Linear Regression model, like we did in the previous chapter:

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

```python
# Let test with some simple data point
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5}
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
```

```python
# Let measure the prediction error of model via RMSE
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predic(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housng_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```

⇒ Okay, this is better than nothing but clearly not a great score: most districts’ median_housing_values range between $120,000 and $265,000, so a typical prediction error of $68,628 is not very satisfying. This is an example of a model under-fitting the training data.

- Let’s train a **`*DecisionTreeRegressor*`**.

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
```
```python
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_remse = np.sqrt(tree_mse)
tree_rmse
```

That’s right: the Decision Tree model is overfitting so badly that it ***performs worse
than the Linear Regression model***.

```python
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared_housing_labels)
```

> 
> 
> 
> Wow, this is much better: Random Forests look very promising. However, note that
> the score on the **`*training set is still much lower than on the validation sets, meaning that the model is still overfitting the training set*`**.

## Fine-Tune Your Model

### Grid Search

> One way to do that would be to fiddle with the hyperparameters manually, until you find a great combination of hyper-parameter values. This would be very tedious work, and you may not have time to explore many combinations.
> 

```python
from sklearn.model_selection import GridSearchCV

param_grid = [
	{'n_estimators': [3. 10. 30], 'max_features': [2, 4, 6, 8]},
	{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
	scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

```

- This param_grid tells Scikit-Learn to first evaluate all 3 × 4 = 12 combinations of n_estimators and max_features hyper-parameter values specified in the first dict, then try all 2 × 3 = 6 combinations of hyper-parameter values in the second dict, but this time with the bootstrap hyper-parameter set to False instead of True

```python
grid_search.best_params_
```

```python
grid_search.best_estimator_
```

```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)
```

### Randomized Search

- The previous model’s parameters fine tuning is just only take experience on a few of possible value, a few combination, but when the hyper-parameter search space is large, it is often preferable to use RandomizedSearchCV instead.
    - If you let the randomized search run for, say, 1,000 iterations, this approach will explore 1,000 different values for each hyper-parameter (instead of just a few values per hyper-parameter with the grid search approach).
    • You have more control over the computing budget you want to allocate to hyper‐parameter search, simply by setting the number of iterations.

### Ensemble Methods

- Another way to fine-tune your system is to try to combine the models that perform best. The group (or “ensemble”) will often perform better than the best individual model (just like Random Forests perform better than the individual Decision Trees they rely on), especially if the individual models make very different types of errors.

### Analyze the Best Models and Their Errors

```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```

```python
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```

### Evaluate Your System on the Test Set
```python
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```

***The performance will usually be slightly worse than what you measured using cross-validation if you did a lot of hyper-parameter tuning*** (because your system ends up fine-tuned to perform well on the validation data, and will likely not perform as well on unknown datasets)

## Launch, Monitor, and Maintain Your System

> You also need to write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops. This is important to catch not only sudden breakage, but also performance degradation. This is quite common because models tend to “rot” as data evolves over time, unless the models are regularly trained on fresh data.
