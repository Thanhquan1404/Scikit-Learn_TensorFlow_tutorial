# The Machine Learning Landscape

## What is Machine Learning?

> ***Machine Learning is the science (and art) of programming computers so they can learn from data.***

  For example, your spam filter is a ***Machine Learning program*** that can ***learn to flag spam given examples of spam emails*** (e.g., flagged by users) and examples of regular (nonspam, also called “ham”) emails. The examples that ***the system uses to learn are called the training set.*** Each training example is called a training instanc (or sample). In this case, the task T is to flag spam for new emails, the experience E is the training data, and the performance measure P needs to be defined; for example, you can use the ratio of correctly classified emails. This particular performance measure is called accuracy and it is often used in classification tasks.
## Why use Machine Learning?

> - ***How to configure the spam filter using traditional programming techniques:***
>     - ***First you would look at what spam typically looks like. You might notice that some words or phrases tend to come up a lot in the subject. Perhaps you would also notice a few other patterns in the sender’s name, the email’s body, and so on.***
>     - ***You would write a detection algorithm for each of the patterns that you noticed, and your program would flag emails as spa, if a number of these patterns are detected.***
>     - ***You would test your program, and repeat steps 1 and 2 until it is good enough.***

>- ***What the difference between traditional and machine learning solution:***
>   - ***if spammers notice that all their emails containing “4U” are blocked, they might start writing “For U” instead. A spam filter using traditional programming techniques would need to be updated to flag “For U” emails. If spammers keep working around your spam filter, you will need to keep writing new rules forever.***
>   - ***Machine learning is a spam filter based on Machine Learning techniques automatically notices that “For U” has become unusually frequent in spam flagged by users, and it starts flagging them without your intervention***

>- ***Machine Learning is great for:***
>     - ***Problems for which existing solutions require a lot of hand-tuning or long list of rules: one Machine Learning algorithm can often simplify code and perform better.***
>     - ***Complex problems for which there is so good solution at all using a traditional approach: the best Machine Learning techniques and find a solution.***
>     - ***Fluctuating environments: a Machine Learning system can adapt to new data***
>     - ***Getting insights about complex problems and large amounts of data***

## Types of Machine Learning Systems
> - ***There are so many different types of Machine Learning systems that it is useful to classify them in broad categories based on:***
>     - ***Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)***
>     - ***Whether or not they can learn incrementally on the fly (online versus batch learning)***
>     - ***Whether they work by simply comparing new data points to known data points, or instead detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning)***

## Supervised/Unsupervised/Semisupervised Learning

> - T***here are four major categories:***
>     - ***Supervised learning: The training data you feed to the algorithm includes the desired solutions, called labels. A typical supervised learning task is classification, another typical task is to predict a target numeric value.***
>         - ***K-Nearest Neighbors.***
>         - ***Linear Regression***
>         - ***Logistic Regression***
>         - ***Support Vector Machine (SVMs)***
>         - ***Decision Trees and Random Forests***
>         - ***Neural networks***
>     - ***Unsupervised learning: In unsupervised learning, as you might guess, the training data is unlabeled. The core output of model is clustering identification:***
>         - ***Clustering:***
>             - ***k-Means***
>             - ***Hierarchical Cluster Analysis (HCA)***
>             - ***Expectation Maximization***
>         - ***Visualization and dimensionality reduction***
>             - ***Principal Component Analysis (PCA)***
>             - ***Kernel PCA***
>             - ***Locally-Linear Embedding (LLE)***
>             - ***t-distributed Stochastic Neighbor Embedding (t-SNE)***
>         - ***Association rule learning***
>             - ***Apriori***
>             - ***Eclat***
>     - ***Semisupervised: Some algorithms can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. This is called semisupervised learning. Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms:***
>         - ***Deep Belief Network (DBNs)***
>         - ***Restricted Boltzmann Machines (RBMs):  are trained sequentially in an unsupervised manner, and then the whole system is fine-tuned using supervised learning techniques.***

> ***Visualization algorithms are also good examples of unsupervised learning algorithm - You feed them a lot of complex and unlabeled data, and they output a 2D or 3D representation of your data that can easily be plotted.***
> 

> ***A related task is dimensionality reduction, in which the goal is to simplify the data
> without losing too much information. One way to do this is to merge several correlated features into one.***

> ***Yet another important unsupervised task is anomaly detection - for example, detecting unusual credit card transactions to prevent fraud, catching manufacturing defects, or automatically removing outliers from a dataset before feeding it to another learning algorithm. The system is trained with normal instances, and when it sees a new instance it can tell whether it looks like a normal one or whether it is likely an anomaly***

> ***Another common unsupervised task is association rule learning, in which the goal is to dig into large amounts of data and discover interesting relations between attributes. For example, suppose you own a supermarket. Running an association rule on your sales logs may reveal that people who purchase barbecue sauce and potato chips also tend to buy steak. Thus, you may want to place these items close to each other.***

## Reinforcement Learning

> ***Reinforcement Learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative reward). It must then learn by itself what is the best strategy, called policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.***

