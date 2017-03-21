import pandas as pd
import numpy as np
import keras
import os

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.embeddings import Embedding

from keras.layers.core import Flatten
from keras.layers import Input, Embedding, LSTM, Dense, merge, Dropout
from keras.optimizers import Adam

from keras.regularizers import l2, activity_l2

path = '/home/pranand/MovieRecommendation/data/ml-latest-small/'
model_path = path + 'models/'

if not os.path.exists(model_path):
	os.mkdir(model_path)

data = pd.read_csv(path + 'ratings.csv')

movie = pd.read_csv(path + 'movies.csv').set_index('movieId')
movie.head()

movie_names = pd.read_csv(path+'movies.csv').set_index('movieId')['title'].to_dict()

users = data.userId.unique()
movies = data.movieId.unique()

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}

data.movieId = data.movieId.apply(lambda x: movieid2idx[x])
data.userId = data.userId.apply(lambda x: userid2idx[x])

user_min, user_max, movie_min, movie_max = (data.userId.min(),
    data.userId.max(), data.movieId.min(), data.movieId.max())


n_users = data.userId.nunique()
n_movies = data.movieId.nunique()

n_factors = 50

np.random.seed = 42

msk = np.random.rand(len(data)) < 0.8
trn = data[msk]
val = data[~msk]

def create_embedding(name, n_in, n_out, reg):
	input = Input(shape=(1,), dtype='int64', name=name)
	return input, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(input)

user_in, u = create_embedding('user_in', n_users, n_factors, 1e-4)
movie_in, m = create_embedding('movie_in', n_movies, n_factors, 1e-4)

# x = merge([u, m], mode='concate')
# x = Flatten()(x)
# x = Dropout(0.3)(x)
# x = Dense(70, 'relu')(x)
# x=Dropout(0.7)(x)
# x = Dense(1)(x)
#
# nn = Model([user_in, movie_in], x)

def model(u, m, user_in, movie_in, learning_rate=0.001):
	x = merge([u, m], mode='concat')
	x = Flatten()(x)
	x = Dropout(0.3)(x) # 0.3
	x = Dense(70, activation='relu')(x)
	x = Dropout(0.75)(x)
	x = Dense(1)(x)
	
	nn = Model([user_in, movie_in], x)
	nn.compile(optimizer=Adam(0.001), loss='mse')
	return nn

nn = model(u, m, user_in, movie_in)
nn.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=8,
          validation_data=([val.userId, val.movieId], val.rating))

