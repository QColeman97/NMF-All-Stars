---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
Quinn Coleman & Andrew Keshishian


## Should we grade this notebook? (Answer yes or no)


yes


**Instructions:** Pair programming assignment. Submit only a single notebook unless you deviate significantly after lab on Thursday. If you submit individually, make sure you indicate who you worked with originally. Make sure to include your first and last names. For those students who push to individual repos but still work in groups, please indicate which notebook should be graded.


# Recommendation Systems

## Lab Assignment

This is a pair programming assignment. I strongly
discourage individual work for this (and other team/pair programming) lab(s), even if you think you can do it
all by yourself. Also, this is a pair programming assignment, not a ”work in teams of two” assignment. Pair
programming requires joint work on all aspects of the project without delegating portions of the work to individual
1
team members. For this lab, I want all your work — discussion, software development, analysis of the results,
report writing — to be products of joint work.
Students enrolled in the class can pair with other students enrolled in the class. Students on the waitlist can
pair with other students on the waitlists. In the cases of ”odd person out” situations, a team of three people can
be formed, but that team must (a) ask and answer one additional question, and (b) work as a pair would, without
delegation of any work off-line.


## At the end of this lab, I should be able to
* Understand how item-item and user-user collaborative filtering perform recommendations
* Explain a experiment where we tested item-item versus user-user

```python
# We need a better version
!pip install -U scikit-learn
```

## Our data
We will be using a well known movielens dataset (small version).


### Here are all the imports that I've used

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
```

```python
ratings = pd.read_csv('~/csc-466-student/data/movielens-small/ratings.csv') # you might need to change this path
```

```python
ratings = ratings.dropna()
ratings
```

```python
len(ratings.userId.unique())
```

```python
movies = pd.read_csv('~/csc-466-student/data/movielens-small/movies.csv')
```

```python
movies = movies.dropna()
movies
```

### Joining the data together
We need to join those two source dataframes into a single one called data. I do this by setting the index to movieId and then specifying an ``inner`` join which means that the movie has to exist on both sides of the join. Then I reset the index so that I can later set the multi-index of userId and movieId. The results of this are displayed below. Pandas is awesome, but it takes some getting used to how everything works.

```python
data = movies.set_index('movieId').join(ratings.set_index('movieId'),how='inner').reset_index()
#data["movieId"] = data["title"]+" "+data["movieId"].astype(str)
data = data.set_index(['userId','movieId'])[["rating"]]
data
```

### Turning data into a matrix instead of a series
The functions ``stack()`` and ``unstack()`` are called multiple times in this lab. They allow me to easily change from a dataframe to a series and back again. Below I'm changing from the Series object to a DataFrame. The important thing to note is that each row is now a user! NaN values are inserted where a user did not rate movie.

```python
data=data.unstack()
data
```

## Let's take a look at some useful code together before the exercises.

First let's look at code that centers the data (important for cosine distance) and then fills in missing values as 0.

```python
data_centered = data-data.mean()
data_centered = data_centered.fillna(0)
data_centered
```

### Now what if we want to grab a specific user? Let's say we want the one with user ID of 1.

```python
x = data_centered.loc[1]
x
```

### Finding neighborhood.
If we are hoping to predict movies for this user, then user-user collaborative filtering says find the ``N`` users that are similar. We should definitely drop out user 1 because it makes no sense to recommend to yourself. We then compute the cosine similarity between this user ``x`` and all other users in the db. We then reverse sort them, and then display the results.

```python
db = data_centered.drop(1)
sims = db.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
sorted_sims = sims.sort_values()[::-1]
sorted_sims
```

### Grabing similar users
Let's set the network size to 10, and then grab those users :)

```python
N=10
userIds = sorted_sims.iloc[:N].index
data_centered.loc[userIds]
```

### How about a prediction?
We could compute the mean from the neighborhood for each prediction

```python
db.loc[userIds]
db.loc[userIds].mean()
# data.loc[1].mean()
# db.loc[userIds].mean()+data.loc[1].mean()
```

### What if we want to weight by the distance?

```python
display(db.loc[userIds].multiply(2))
display(db.loc[userIds].multiply(2, axis=0))
db.loc[userIds].multiply(2, axis=1)

db.loc[userIds].multiply(sorted_sims.iloc[:N],axis=0).sum()/sorted_sims.iloc[:N].sum()+data.loc[1].mean()
```

## Finally to the exercises!
I want you to implement user-user, item-item, and a combination of item-item and user-user.

```python
data
```

## Exercise 1 (Worth 5 points)
Complete the following function that predicts using user-user collaborative filtering. 

```python
def predict_user_user(data_raw,x_raw,N=10,frac=0.02):
    # data_raw is our uncentered data matrix. We want to make sure we drop the name of the user we
    # are predicting:
    db = data_raw.drop(x_raw.name)
    # We of course want to center and fill in missing values
    db = (db-db.mean()).fillna(0)
    # Now this is a little tricky to think about, but we want to create a train test split of the movies
    # that user x_raw.name has rated. We need some of them but want some of them removed for testing.
    # This is where the frac parameter is used. I want you to think about how to select movies for training
    # ix_raw, ix_raw_test = train_test_split(???,test_size=frac,random_state=42) # Got to ignore some movies
    
    # Filter out movies that user x hasn't rated
    # Find all movies user x has rated
    
    ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42)
    # Here is where we use what you figured out above
    x_raw_test = x_raw.loc[ix_raw_test]
    x_raw = x_raw.copy()
    x_raw.loc[ix_raw_test] = np.NaN # ignore the movies in test
    x = (x_raw - x_raw.mean()).fillna(0)

    preds = []
    for movie in ix_raw_test:
        sims = db.loc[data_raw.drop(x_raw.name)[movie].isnull()==False].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        #sims = db.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        try:
            sorted_sims = sims.sort_values()[::-1]
        except:
            preds.append(0) # means there is no one that also rated this movie amongst all other users
            continue
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        preds.append(db.loc[ids][movie].mean())
        #preds.append(x_raw[ids])
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test-x_raw.mean()
    mae = (actual-pred).abs().mean()
    return mae
```

```python
data.loc[1]

```

```python
mae = predict_user_user(data,data.loc[1])
mae
```

```python
maes = data.head(20).apply(lambda x: predict_user_user(data,x),axis=1)
```

```python
np.mean(maes)
```

## Exercise 2 (Worth 5 points)
Complete the following function that predicts using item-item collaborative filtering. 

```python
def predict_item_item(data_raw,x_raw,N=10,frac=0.02,debug={}):
    # x_raw is a user (row)
    ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42) # Got to ignore some movies
    
    print('ix_raw and ix_raw_test:')
    print(ix_raw, ix_raw_test)
    
    # Indices of movies our user has seen
    x_raw_test = x_raw.loc[ix_raw_test]
    
    print('x_raw_test:')
    print(x_raw_test)
    
    db = data_raw.drop(x_raw.name)
    db = (db-db.mean()).fillna(0)
    # ??? db = FIX DB SO WE CAN KEEP CODE SIMILAR BUT DO ITEM-ITEM ???
    db = db.T # Columns are all users except ours, rows are all movies
    
    preds = []
    for movie in ix_raw_test:
        x = db.loc[movie]
        # x is a row of db, all users except ours who have also rated this movie we're testing
        
        print('Movie:', movie)
        
#         breakpoint()
#         sims = db.drop(movie).loc[??? ONLY SELECT MOVIES IN TRAINING SET WHICH USER HAS RATED ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
#         sims = db.drop(movie).loc[x_raw.drop(x_raw.name)[movie].isnull()==False].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = db.drop(movie).loc[data_raw.T[x_raw.name].isnull()==False].apply(
            lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        
        print('Sims:', sims)
        
        # db has movie rows except movie row we're testing, where our user rated this movie
        
        sims = sims.dropna()
        sorted_sims = sims.sort_values()[::-1]
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? HOW TO PREDICTION ???)
#         breakpoint()
        preds.append((data_raw-data_raw.mean()).fillna(0).T.loc[ids][x_raw.name].mean())
    
        # Pred is the avg of movie ratings of this user, 
        # where the movies are the ones in the neighborhood
    
#         preds.append(db.loc[ids].mean().mean())
        # Prediction is the avg of the ratings of most similar movies
        
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test
    mae = (actual-pred).abs().mean()
    return mae

# def predict_user_user(data_raw,x_raw,N=10,frac=0.02):
#     # x_raw is a user
#     db = data_raw.drop(x_raw.name)
#     db = (db-db.mean()).fillna(0)
#     # db has no user x_raw (removed row)
    
#     ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42)
#     # Indices of movies x_raw has seen
#     x_raw_test = x_raw.loc[ix_raw_test]
#     # (Test) movies x_raw seen
#     x_raw = x_raw.copy()
#     x_raw.loc[ix_raw_test] = np.NaN # ignore the movies in test
#     # x_raw is moves x_raw ignoring test movies
#     x = (x_raw - x_raw.mean()).fillna(0)

#     preds = []
#     for movie in ix_raw_test:
#         # movie is an index of seen movie by user
#         sims = db.loc[data_raw.drop(x_raw.name)[movie].isnull()==False].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
#         # Find all the people who have seen the movie seen by user, compute similarities
#         sims = sims.dropna()
#         try:
#             sorted_sims = sims.sort_values()[::-1]
#         except:
#             preds.append(0) # means there is no one that also rated this movie amongst all other users
#             continue
#         top_sims = sorted_sims.iloc[:N]
#         ids = top_sims.index
#         preds.append(db.loc[ids][movie].mean())
#         # Prediction is the avg of the ratings of most similar users
#     pred = pd.Series(preds,index=x_raw_test.index)
#     actual = x_raw_test-x_raw.mean()
#     mae = (actual-pred).abs().mean()
#     return mae
```

```python
data_dict = {('rating', "A"): [3, np.nan, 1, 5],
             ("rating", "B"): [np.nan, 4, 3, np.nan],
             ("rating", "C"): [2, np.nan, 2, 3], 
             ("rating", "D"): [np.nan, 3, np.nan, np.nan]}
data = pd.DataFrame(data_dict)
display(data)

mae = predict_item_item(data,data.loc[0])
mae
```

```python
maes = data.head(20).apply(lambda x: predict_item_item(data,x),axis=1)
```

```python
np.mean(maes)
```

**For this very simple experiment, what method seems better?**

YOUR ANSWER HERE


## Exercise 3 (Worth 5 points)
Create new versions of predict_user_user and predict_item_item, but now perform a weighted prediction as was demonstrated above. Did our accuracy get any better?

```python
# Weighted Avg
# db.loc[userIds].multiply(sorted_sims.iloc[:N],axis=0).sum()/sorted_sims.iloc[:N].sum()+data.loc[1].mean()

def predict_item_item(data_raw,x_raw,N=10,frac=0.02,debug={}):
    ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42) # Got to ignore some movies
    x_raw_test = x_raw.loc[ix_raw_test]
    
    db = data_raw.drop(x_raw.name)
    db = (db-db.mean()).fillna(0)
    # ??? db = FIX DB SO WE CAN KEEP CODE SIMILAR BUT DO ITEM-ITEM ???
    db = db.T
    
    preds = []
    for movie in ix_raw_test:
        x = db.loc[movie]
        # sims = db.drop(movie).loc[??? ONLY SELECT MOVIES IN TRAINING SET WHICH USER HAS RATED ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = db.drop(movie).loc[data_raw.T[x_raw.name].isnull()==False].apply(
            lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        sorted_sims = sims.sort_values()[::-1]
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? HOW TO PREDICTION ???)
        preds.append((data_raw-data_raw.mean()).fillna(0).T.loc[ids][x_raw.name]
                     .multiply(top_sims,axis=0).sum()/top_sims.sum())

    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test
    mae = (actual-pred).abs().mean()
    return mae

def predict_user_user(data_raw,x_raw,N=10,frac=0.02):
    # data_raw is our uncentered data matrix. We want to make sure we drop the name of the user we
    # are predicting:
    db = data_raw.drop(x_raw.name)
    # We of course want to center and fill in missing values
    db = (db-db.mean()).fillna(0)
    # Now this is a little tricky to think about, but we want to create a train test split of the movies
    # that user x_raw.name has rated. We need some of them but want some of them removed for testing.
    # This is where the frac parameter is used. I want you to think about how to select movies for training
    #ix_raw, ix_raw_test = train_test_split(???,test_size=frac,random_state=42) # Got to ignore some movies
    ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42)

    # Here is where we use what you figured out above
    x_raw_test = x_raw.loc[ix_raw_test] 
    x_raw = x_raw.copy()
    x_raw.loc[ix_raw_test] = np.NaN # ignore the movies in test
    x = (x_raw - x_raw.mean()).fillna(0)

    preds = []
    for movie in ix_raw_test:
        #sims = db.loc[??? Only look at users who have rated this movie ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = db.loc[data_raw.drop(x_raw.name)[movie].isnull()==False].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        try:
            sorted_sims = sims.sort_values()[::-1]
        except:
            preds.append(0) # means there is no one that also rated this movie amongst all other users
            continue
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? using ids how do you predict ???)
        preds.append(db.loc[ids][movie].multiply(top_sims,axis=0).sum()/top_sims.sum())
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test-x_raw.mean()
    mae = (actual-pred).abs().mean()
    return mae
```

```python
mae = predict_item_item(data,data.loc[1])
mae
```

```python
mae = predict_user_user(data,data.loc[1])
mae
```

## Exercise 4 (Worth 5-10 extra credit points for one or both implementions)
Combine in sequence item-item and user-user AND/OR user-user and item-item.

```python

```
