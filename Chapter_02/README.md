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