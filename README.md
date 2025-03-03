# RECOMMENDATION-SYSTEM

COMPANY NAME: CODETECH IT SOLUTIONS

NAME: NITIN MAHOR

ITERN ID: CT08RWJ

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH KUMAR

### **Building a Collaborative Filtering-Based Recommendation System**

#### **Introduction**
In today’s digital landscape, recommendation systems play a crucial role in enhancing user experiences on platforms like Netflix, Amazon, and Spotify. A recommendation system suggests relevant items to users by analyzing their past interactions, preferences, and behaviors. One of the most common approaches is **collaborative filtering**, which relies on similarities between users or items to make predictions.

This document describes the implementation of a **user-based collaborative filtering** recommendation system using Python. The system predicts item recommendations based on the preferences of users who share similar interests. The implementation leverages the **cosine similarity metric** to measure user-user relationships and generate personalized suggestions.

---

### **Understanding Collaborative Filtering**
Collaborative filtering is a technique that recommends items based on the actions of similar users. It is divided into two main types:

1. **User-Based Collaborative Filtering**:  
   - Finds users who have similar tastes and recommends items based on their preferences.
   - Example: If User A and User B both like "The Dark Knight," and User A also likes "Inception," then User B might be recommended "Inception."

2. **Item-Based Collaborative Filtering**:  
   - Finds similarities between items instead of users.
   - Example: If many users who bought a laptop also bought a mouse, then a new laptop buyer may be recommended a mouse.

This implementation focuses on **user-based collaborative filtering** using **cosine similarity**.

---

### **Code Breakdown**
Below is a detailed explanation of the implementation:

#### **1. Importing Required Libraries**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
```
- **NumPy (`numpy`)**: Used for numerical operations and handling matrices efficiently.
- **Pandas (`pandas`)**: Used for organizing the dataset in tabular form.
- **Scikit-learn (`sklearn`)**: Provides tools for computing similarity metrics like **cosine similarity**.

---

#### **2. Creating a Sample Dataset**
```python
data = {
    "User": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
              6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10],
    "Item": [1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 4, 5, 1, 3, 4, 5, 2, 3, 4, 5,
              1, 2, 3, 4, 1, 3, 4, 5, 2, 3, 4, 5, 1, 2, 3, 5, 2, 3, 4, 5],
    "Rating": [5, 3, 4, 2, 4, 2, 5, 3, 3, 5, 4, 4, 5, 3, 4, 2, 1, 5, 3, 4,
                4, 5, 3, 2, 5, 2, 4, 3, 3, 5, 4, 5, 4, 3, 5, 2, 1, 4, 5, 3]
}
df = pd.DataFrame(data)
```
- This dataset contains **users, items (products/movies), and ratings**.
- Each user provides ratings for different items on a scale of **1 to 5**.

---

#### **3. Creating a User-Item Matrix**
```python
user_item_matrix = df.pivot(index="User", columns="Item", values="Rating").fillna(0)
```
- This converts the dataset into a **user-item matrix** where:
  - **Rows** represent users.
  - **Columns** represent items.
  - **Cells** store ratings given by users to items.
  - Missing values (where a user hasn’t rated an item) are filled with `0`.

---

#### **4. Computing User Similarity Using Cosine Similarity**
```python
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
```
- **Cosine Similarity** measures the similarity between users based on their rating patterns.
- The result is a **similarity matrix** where each cell `(i, j)` contains the similarity score between User `i` and User `j`.

---

#### **5. Building the Recommendation Function**
```python
def recommend(user_id, num_recommendations=3):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    user_ratings = user_item_matrix.loc[user_id]
    recommendations = {}
    
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        for item in user_item_matrix.columns:
            if user_ratings[item] == 0 and similar_user_ratings[item] > 0:
                recommendations[item] = recommendations.get(item, 0) + similar_user_ratings[item]
    
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [item for item, rating in sorted_recommendations[:num_recommendations]]
```
- **Identifies similar users** based on cosine similarity.
- **Finds unrated items** for the target user that similar users have rated.
- **Aggregates and ranks items** to recommend the most relevant ones.

---

#### **6. Generating Recommendations**
```python
for user_id in user_item_matrix.index:
    print(f"Recommended items for User {user_id}: {recommend(user_id)}")
```
- This loops through all users and generates personalized recommendations.

---

### **Tools and Platform Used**
#### **1. Tools Used**
- **Python**: Programming language.
- **NumPy**: Efficient matrix operations.
- **Pandas**: Data handling.
- **Scikit-learn**: Cosine similarity computation.

#### **2. Platform**
- **Jupyter Notebook**: Used for running the script interactively.

---

### **How to Run the Code**
1. **Install dependencies**:
   ```bash
   pip install numpy pandas scikit-learn
   ```
2. **Run the script in Jupyter Notebook**.

---

### **Conclusion**
This **user-based collaborative filtering system** recommends items by analyzing user similarity. While this approach is simple and effective, it has **scalability limitations** for large datasets. More advanced techniques, such as **matrix factorization** or **deep learning-based recommendation systems**, can be explored for better accuracy and efficiency.


#OUTPUT

![Image](https://github.com/user-attachments/assets/8584e082-a567-4326-ba41-7bfbe2f90af6)
