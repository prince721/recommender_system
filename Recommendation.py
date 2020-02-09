# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:32:10 2020

@author: Prince Mishra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import seaborn as snb


columns=['user_id','item_id','rating','timestamp']
url='https://raw.githubusercontent.com/krishnaik06/Movie-Recommender-in-python/master/u.data'
df=pd.read_csv(url,sep='\t',names=columns)


m_url='https://raw.githubusercontent.com/krishnaik06/Movie-Recommender-in-python/master/Movie_Id_Titles'
movie_titles=pd.read_csv(m_url)

df=pd.merge(df,movie_titles,on='item_id')
df.head()

df.describe()


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

df.groupby('title')['rating'].count().sort_values(ascending=False).head()


ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['number_of_ratings'] =df.groupby('title')['rating'].count()

ratings['rating'].hist(bins=50)
ratings['number_of_ratings'].hist(bins=50)

snb.jointplot(x='rating',y='number_of_ratings',data=ratings)

movie_matrix=df.pivot_table(index='user_id',columns='title',values='rating')


ratings.sort_values('number_of_ratings',ascending=False).head()


starwars_user_ratings = movie_matrix['Star Wars (1977)']
liarliar_user_ratings = movie_matrix['Liar Liar (1997)']
starwars_user_ratings.head()

similar_to_starwars=movie_matrix.corrwith(starwars_user_ratings)

corr_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)


#corr_starwars.sort_values('Correlation',ascending=False).head()

corr_starwars=corr_starwars.join(ratings['number_of_ratings'])
corr_starwars[corr_starwars['number_of_ratings']>100].sort_values('Correlation',ascending=False).head()

