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