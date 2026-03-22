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