### **1. How would you define Machine Learning?**

Machine Learning is a field of AI where systems learn patterns from data and improve performance on a task without being explicitly programmed.


### **2. Four types of problems where it shines**

* Image recognition
* Natural language processing
* Recommendation systems
* Fraud detection


### **3. What is a labeled training set?**

A dataset where each input is paired with the correct output (label).


### **4. Two most common supervised tasks**

* Classification
* Regression


### **5. Four common unsupervised tasks**

* Clustering
* Dimensionality reduction
* Anomaly detection
* Association rule learning


### **6. Robot walking in unknown terrains → which algorithm?**

👉 **Reinforcement Learning** (learning by trial and error with rewards)


### **7. Segment customers into groups → which algorithm?**

👉 **Clustering (Unsupervised Learning)**


### **8. Spam detection → supervised or unsupervised?**

👉 **Supervised learning** (emails labeled as spam/not spam)


### **9. What is an online learning system?**

A system that learns incrementally from data as it arrives (streaming data).


### **10. What is out-of-core learning?**

Learning from data that is too large to fit into memory, using batches.


### **11. Algorithm relying on similarity measure?**

👉 **Instance-based learning** (e.g., k-Nearest Neighbors)


### **12. Model parameter vs hyperparameter**

* **Parameter**: Learned from data (e.g., weights in linear regression)
* **Hyperparameter**: Set before training (e.g., learning rate, k in k-NN)


### **13. Model-based learning algorithms**

* They search for a **model that best fits the data**
* Strategy: **Minimize a cost/loss function**
* Predictions: Apply the learned model to new data


### **14. Four main challenges in Machine Learning**

* Insufficient data
* Poor-quality data
* Overfitting
* Underfitting


### **15. Good on training but poor on new data → what happens?**

👉 **Overfitting**

**Three solutions:**

* Get more training data
* Simplify the model
* Use regularization


### **16. What is a test set? Why use it?**

A separate dataset used to evaluate final model performance on unseen data.


### **17. Purpose of a validation set**

Used to tune hyperparameters and compare models during training.


### **18. What goes wrong if you tune using test set?**

👉 You **overfit the test set**, giving overly optimistic performance estimates.


### **19. What is cross-validation? Why prefer it?**

A technique where data is split into multiple folds and trained/evaluated multiple times.

👉 Preferred because:

* Uses data more efficiently
* Gives more reliable performance estimates
