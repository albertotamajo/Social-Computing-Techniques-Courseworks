import numpy as np
import itertools
from collections.abc import Iterable
import argparse


# --------------------------LOADING CSV FUNCTIONS------------------------------#
def load_data(file):
    """
    Load a csv file and convert it into a numpy array.
    :param file: csv file path
    :return: numpy array representation of the csv file
    """
    with open(file) as f:
        lines = f.readlines()
    return np.asarray([line.strip('\n').split(',') for line in lines])


# --------------------------SAVING MATRIX TO FILE FUNCTIONS---------------------#


def save_matrix(file, x, delimiter, fmt):
    """
    Save a numpy array into a file with a given delimiter and formatting
    :param file: file path where to save the numpy array
    :param x: numpy array to be saved
    :param delimiter: delimiter used between columns in the csv file
    :param fmt: formatting
    :return None
    """
    np.savetxt(file, x, delimiter=delimiter, fmt=fmt)


def save_matrix_csv(file, x, fmt):
    """
    Save a numpy array into a file with a given formatting and ',' as delimiter
    :param file: file path where to save the numpy array
    :param x: numpy array to be saved
    :param fmt: formatting
    :return: None
    """
    save_matrix(file, x, ',', fmt)


def save_prediction_matrix(file, x):
    """
    Save a numpy array into a file with ',' as delimiter and string syntax formatting
    :param file: file path where to save the numpy array
    :param x: numpy array to be saved
    :return: None
    """
    save_matrix_csv(file, x, '%s')


# -----------------------------MATRIX FUNCTIONS---------------------------------#

def rating_matrix(lines):
    """
    Build a (n x i) rating matrix from a numpy array containing user ratings.
    n is the number of users, i is the number of items.
    :param lines: a numpy array containing user ratings
    :return: (n x i) rating matrix, where n is the number of users and i the number of items.
    """
    n_users, n_items = lines[:, :2].astype(int).max(axis=0)
    rating_matrix = np.zeros((n_users, n_items), dtype=np.float16)
    for user, item, rating, _ in lines:
        rating_matrix[int(user) - 1, int(item) - 1] = rating
    return rating_matrix


def mean_center_rating_matrix(rating_matrix):
    """
    Mean-center rating matrix.
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items.
    :return: mean-centred rating matrix
    """
    means = np.apply_along_axis(mean_rating, 1, rating_matrix).reshape((rating_matrix.shape[0], 1))
    indx = np.where(rating_matrix == 0)
    rating_matrix = rating_matrix - means
    rating_matrix[indx] = 0
    return rating_matrix


def timestamp_matrix(lines):
    """
    Build a (n x i) timestamp matrix from a numpy array containing user ratings with their timestamps.
    n is the number of users, i is the number of items.
    :param lines: a numpy array containing user ratings
    :return: (n x i) timestamp matrix, where n is the number of users and i the number of items.
    """
    n_users, n_items = lines[:, :2].astype(int).max(axis=0)
    timestamp_matrix = np.zeros((n_users, n_items), dtype=int)
    for user, item, _, timestamp in lines:
        timestamp_matrix[int(user) - 1, int(item) - 1] = timestamp
    return timestamp_matrix


def user_based_prediction_matrix(lines, rating_matrix, similarity_matrix, neighborhood_f, prediction_f):
    """
    Compute the user based rating predictions contained in a numpy array. Substantially, this function takes a numpy array
    representing a csv test file and returns the same numpy array but with an additional column for the predictions. The
    number of nan predictions is also returned. When a nan prediction is made, the prediction is remade using the 10
    closest neighbors of the user in question. If the result is nan again, then the average user rating is used as rating
    prediction.
    :param lines: numpy array containing predictions to be computed
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items.
    :param similarity_matrix: (n x n) similarity matrix, where n is the number of users
    :param neighborhood_f: neighborhood function/s. A single neighborhood function can be provided or a list of
                           neighborhood functions, one for each prediction to be made
    :param prediction_f: user based prediction function
    :return: rating prediction matrix, nan predictions
    """
    nan_predictions = 0
    predictions = []
    iter = isinstance(neighborhood_f, Iterable)
    for n, (user, item, _) in enumerate(lines):
        user = int(user) - 1
        item = int(item) - 1
        pred = None
        if iter:
            pred = user_based_user_item_prediction(user, item, rating_matrix, similarity_matrix[user], neighborhood_f[n], prediction_f)
        else:
            pred = user_based_user_item_prediction(user, item, rating_matrix, similarity_matrix[user], neighborhood_f, prediction_f)
        if np.isnan(pred):
            nan_predictions += 1
            pred = user_based_user_item_prediction(user, item, rating_matrix, similarity_matrix[user], k_neighborhood(10), prediction_f)
            if np.isnan(pred):
                pred = mean_rating(rating_matrix[user])
                if np.isnan(pred):
                    pred = 2.5
        predictions.append(pred)
    predictions = np.asarray(predictions, dtype=np.str)
    predictions = np.expand_dims(predictions, axis=1)
    return np.append(np.append(lines[:, :2], predictions, axis=1), np.expand_dims(lines[:, 2], axis=1), axis=1), nan_predictions


def item_based_prediction_matrix(lines, rating_matrix, similarity_matrix, neighborhood_f, prediction_f):
    """
    Compute the item based rating predictions contained in a numpy array. Substantially, this function takes a numpy array
    representing a csv test file and returns the same numpy array but with an additional column for the predictions. The
    number of nan predictions is also returned. When a nan prediction is made, the prediction is remade using the 10
    closest neighbors of the item in question. If the result is nan again, then the average user rating is used as rating
    prediction.
    :param lines: numpy array containing predictions to be computed
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items.
    :param similarity_matrix: (i x i) similarity matrix, where i is the number of items
    :param neighborhood_f: neighborhood function/s. A single neighborhood function can be provided or a list of
                           neighborhood functions, one for each prediction to be made
    :param prediction_f: user based prediction function
    :return: rating prediction matrix, nan predictions
    """
    nan_predictions = 0
    predictions = []
    iter = isinstance(neighborhood_f, Iterable)
    for n, (user, item, _) in enumerate(lines):
        user = int(user) - 1
        item = int(item) - 1
        pred = None
        if iter:
            pred = item_based_user_item_prediction(user, rating_matrix, similarity_matrix[item], neighborhood_f[n], prediction_f)
        else:
            pred = item_based_user_item_prediction(user, rating_matrix, similarity_matrix[item], neighborhood_f, prediction_f)
        if np.isnan(pred):
            nan_predictions += 1
            pred = mean_rating(rating_matrix[user])
            if np.isnan(pred):
                pred = 2.5
        predictions.append(pred)
    predictions = np.asarray(predictions, dtype=np.str)
    predictions = np.expand_dims(predictions, axis=1)
    return np.append(np.append(lines[:, :2], predictions, axis=1), np.expand_dims(lines[:, 2], axis=1), axis=1), nan_predictions


def user_similarity_matrix(rating_matrix, similarity_f):
    """
    Build a (n x n) user similarity matrix from a (n x i) rating matrix with a given similarity function.
    n is the number of users, i is the number of items.
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items.
    :param similarity_f: similarity function
    :return: (n x n) user similarity matrix, where n is the number of users
    """
    n_users = rating_matrix.shape[0]
    similarity_matrix = np.empty((n_users, n_users))
    for u1, u2 in itertools.combinations(range(n_users), 2):
        sim = similarity_f(rating_matrix[u1], rating_matrix[u2])
        similarity_matrix[u1, u2] = sim
        similarity_matrix[u2, u1] = sim
    for u in range(n_users):
        similarity_matrix[u, u] = np.NaN
    return similarity_matrix


def item_similarity_matrix(rating_matrix, similarity_f):
    """
    Build a (i x i) item similarity matrix from a (n x i) rating matrix with a given similarity function.
    n is the number of users, i is the number of items.
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items.
    :param similarity_f: similarity function
    :return: (i x i) item similarity matrix, where i is the number of items
    """
    n_items = rating_matrix.shape[1]
    similarity_matrix = np.empty((n_items, n_items))
    for i1, i2 in itertools.combinations(range(n_items), 2):
        sim = similarity_f(rating_matrix[:, i1], rating_matrix[:, i2])
        similarity_matrix[i1, i2] = sim
        similarity_matrix[i2, i1] = sim
    for i in range(n_items):
        similarity_matrix[i, i] = np.NaN
    return similarity_matrix

#-----------------------------WEIGHTING FUNCTIONS------------------------#


def inverse_user_frequency(items, rating_matrix):
    """
    Compute the inverse user frequency for the given items.
    :param items: numpy array containing items
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items.
    :return: numpy array representing the inverse user frequency values for the given items
    """
    n_users = rating_matrix.shape[0]
    return np.log(n_users / np.count_nonzero(rating_matrix[:, items], axis=0))


# ----------------------------SIMILARITY MEASURES------------------------#


def rating_intersection(u1_rating, u2_rating):
    """
    Find the items for which both user 1 and user 2 have a rating.
    :param u1_rating: numpy array representing the ratings for user 1
    :param u2_rating: numpy array representing the ratings for user 2
    :return: numpy array representing the items for which both user 1 and user 2 have a rating
    """
    return np.intersect1d(u1_rating.nonzero(), u2_rating.nonzero())


def mean_rating(u_rating):
    """
    Compute the mean rating for a user.
    :param u_rating: numpy array representing the ratings for a user
    :return: mean rating
    """
    return u_rating[u_rating.nonzero()].mean()


def pearson1(u1_rating, u2_rating, w=None):
    """
    Compute the Pearson1 similarity measure between two user ratings. Function based on Equation (2.2) from "Recommender
    Systems - The textbook, Charu C. Aggarwal" with the mean ratings computed only over the items that are rated by both
    users.
    :param u1_rating: numpy array representing the ratings of user 1
    :param u2_rating: numpy array representing the ratings of user 2
    :param w: numpy weight vector. If None, no weight is used during the computation of the similarity measure.
    :return: similarity measure
    """
    intersection = rating_intersection(u1_rating, u2_rating)
    if len(intersection) == 0:
        return np.NaN
    else:
        if w is None:
            w = np.ones(intersection.size)
        u1_rating = u1_rating[intersection]
        u2_rating = u2_rating[intersection]
        u1_rating_mean = mean_rating(u1_rating)
        u2_rating_mean = mean_rating(u2_rating)
        u1_rating = u1_rating - u1_rating_mean
        u2_rating = u2_rating - u2_rating_mean
        return ((w * u1_rating) @ u2_rating) / \
               (np.sqrt((w * (u1_rating ** 2)).sum()) * np.sqrt((w * (u2_rating ** 2)).sum()))


def pearson2(u1_rating, u2_rating, w=None):
    """
    Compute the Pearson2 similarity measure between two user ratings. Function based on Equation (2.2) from "Recommender
    Systems - The textbook, Charu C. Aggarwal" with the mean ratings computed over all the items that are rated.
    :param u1_rating: numpy array representing the ratings of user 1
    :param u2_rating: numpy array representing the ratings of user 2
    :param w: numpy weight vector. If None, no weight is used during the computation of the similarity measure
    :return: similarity measure
    """
    intersection = rating_intersection(u1_rating, u2_rating)
    if len(intersection) == 0:
        return np.NaN
    else:
        u1_rating_mean = mean_rating(u1_rating)
        u2_rating_mean = mean_rating(u2_rating)
        if w is None:
            w = np.ones(intersection.size)
        u1_rating = u1_rating[intersection] - u1_rating_mean
        u2_rating = u2_rating[intersection] - u2_rating_mean
        return ((w * u1_rating) @ u2_rating) / \
               (np.sqrt((w * (u1_rating ** 2)).sum()) * np.sqrt((w * (u2_rating ** 2)).sum()))


def raw_cosine1(u1_rating, u2_rating, w=None):
    """
    Compute the RawCosine1 similarity measure between two user ratings. Function based on Equation (2.5) from
    "Recommender Systems - The textbook, Charu C. Aggarwal".
    :param u1_rating: numpy array representing the ratings of user 1
    :param u2_rating: numpy array representing the ratings of user 2
    :param w: numpy weight vector. If None, no weight is used during the computation of the similarity measure
    :return: similarity measure
    """
    intersection = rating_intersection(u1_rating, u2_rating)
    if len(intersection) == 0:
        return np.NaN
    else:
        if w is None:
            w = np.ones(intersection.size)
        u1_rating = u1_rating[intersection]
        u2_rating = u2_rating[intersection]
        return ((w * u1_rating) @ u2_rating) / \
               (np.sqrt((w * (u1_rating ** 2)).sum()) * np.sqrt((w * (u2_rating ** 2)).sum()))


def raw_cosine2(u1_rating, u2_rating, w=None):
    """
    Compute the RawCosine2 similarity measure between two user ratings. Function based on Equation (2.6) from
    "Recommender Systems - The textbook, Charu C. Aggarwal".
    :param u1_rating: numpy array representing the ratings of user 1
    :param u2_rating: numpy array representing the ratings of user 2
    :param w: numpy weight vector. If None, no weight is used during the computation of the similarity measure
    :return: similarity measure
    """
    intersection = rating_intersection(u1_rating, u2_rating)
    if len(intersection) == 0:
        return np.NaN
    else:
        if w is None:
            w = np.ones(intersection.size)
        u1_rating_intersect = u1_rating[intersection]
        u2_rating_intersect = u2_rating[intersection]
        return ((w * u1_rating_intersect) @ u2_rating_intersect) / \
               (np.sqrt((w * (u1_rating ** 2)).sum()) * np.sqrt((w * (u2_rating ** 2)).sum()))


def adjusted_cosine(i1_rating, i2_rating, w=None):
    """
    Compute the AdjustedCosine similarity measure between two item ratings. Function based on Equation (2.14) from
    "Recommender Systems - The textbook, Charu C. Aggarwal".
    :param i1_rating: numpy array representing the ratings for item  1
    :param i2_rating: numpy array representing the ratings for item 2
    :param w: numpy weight vector. If None, no weight is used during the computation of the similarity measure
    :return: similarity measure
    """
    intersection = rating_intersection(i1_rating, i2_rating)
    if len(intersection) == 0:
        return np.NaN
    else:
        if w is None:
            w = np.ones(intersection.size)
        i1_rating = i1_rating[intersection]
        i2_rating = i2_rating[intersection]
        return ((w * i1_rating) @ i2_rating) / (np.sqrt((w * (i1_rating**2)).sum()) * np.sqrt((w * (i2_rating**2)).sum()))


def discounted_similarity(similarity_f, threshold):
    """
    Compute the similarity measure between two users ratings with the given similarity function and discount it when
    the number of common ratings between the two users is less than a particular threshold. Function based on
    Equation (2.7) from "Recommender Systems - The textbook, Charu C. Aggarwal".
    :param similarity_f: similarity function
    :param threshold: threshold
    :return: function that takes as input two user ratings and a weight vector to compute the similarity measure between
             the two users. The similarity measure is discounted when the number of common ratings between the two users
             is less than a particular threshold.
    """
    def wrapper(u1_rating, u2_rating, w=None):
        inters_size = len(rating_intersection(u1_rating, u2_rating))
        sim = similarity_f(u1_rating, u2_rating, w)
        if inters_size < threshold:
            sim = sim * (inters_size / threshold)
        return sim
    return wrapper


def exponential_similarity(similarity_f, exp):
    """
    Compute the similarity measure between two user ratings with the given similarity function and elevate it to the
    given exponent. Function based on Equation (2.11) from "Recommender Systems - The textbook, Charu C. Aggarwal".
    :param similarity_f: similarity function
    :param exp: exponent
    :return: function that takes as input two user ratings and a weight vector to compute the similarity measure between
             the two users. The similarity measure is elevated to the given exponent.
    """
    def wrapper(u1_rating, u2_rating, w=None):
        return similarity_f(u1_rating, u2_rating, w) ** exp
    return wrapper


def inverse_user_frequency_similarity(similarity_f, rating_matrix):
    """
    Compute the similarity measure between two users ratings using the inverse user frequency values as weights.
    Function based on Equation (2.12) from "Recommender Systems - The textbook, Charu C. Aggarwal".
    :param similarity_f: similarity function
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items.
    :return: function that takes as input two user ratings to compute the similarity measure between
             the two users. The similarity measure is computed using the inverse user frequency values as weights.
    """
    def wrapper(u1_rating, u2_rating):
        intersection = rating_intersection(u1_rating, u2_rating)
        if len(intersection) == 0:
            return np.NaN
        else:
            return similarity_f(u1_rating, u2_rating, w=inverse_user_frequency(intersection, rating_matrix))
    return wrapper


# --------------------------NEIGHBORHOOD FUNCTIONS-----------------------#

def full_neighborhood(u_similarity):
    """
    Find all the neighbors of a given user sorted into ascending order with respect to their distance.
    :param u_similarity: numpy array representing the similarity of a user with respect to all the other users
    :return: numpy array representing all neighbors of a given user sorted into ascending order with respect
             to their distance
    """
    return np.flip(np.argsort(u_similarity))[np.argwhere(np.isnan(u_similarity)).flatten().size:]


def k_neighborhood(k):
    """
    Find the k closest neighbors of a given user.
    :param k: number of closest neighbors
    :return: function that takes as input the similarity vector of a user with respect to all the other users and
             finds the k closest neighbors.
    """
    def wrapper(u_similarity):
        return full_neighborhood(u_similarity)[:k]

    return wrapper


def threshold_neighborhood(t, neighborhood_f):
    """
    Find the neighbors of a given user with the given neighborhood function and discard those whose similarity is less
    than the given threshold.
    :param t: threshold
    :param neighborhood_f: neighborhood function
    :return: function that takes as input the similarity vector of a user with respect to all the other users and
             finds the neighbors of a given user with the given neighborhood function and discard those whose similarity
             is less than the given threshold.
    """
    def wrapper(u_similarity):
        neighborhood = neighborhood_f(u_similarity)
        return neighborhood[(u_similarity[neighborhood] >= t)]
    return wrapper


def item_neighborhood(rating_matrix, item, neighborhood_f):
    """
    Find the neighbors of a user with a given neighborhood function such that the neighbors have a rating
    for the given item.
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items
    :param item: item
    :param neighborhood_f: neighborhood function
    :return: function that takes as input the similarity vector of a user with respect to all the other users
             and finds the neighbors of a user with a given neighborhood function such that the neighbors have a rating
             for the given item.
    """
    def wrapper(u_similarity):
        u_similarity = np.copy(u_similarity)
        for i in range(u_similarity.size):
            if not np.isnan(u_similarity[i]):
                if rating_matrix[i, item] == 0:
                    u_similarity[i] = np.NaN
        return neighborhood_f(u_similarity)
    return wrapper


def user_neighborhood(rating_matrix, user, neighborhood_f):
    """
    Find the neighbors of an item with a given neighborhood function such that the neighbors have a rating
    from a given user.
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items
    :param user: user
    :param neighborhood_f: neighborhood function
    :return: function that takes as input the similarity vector of a user with respect to all the other users and
             finds the neighbors of an item with a given neighborhood function such that the neighbors have a rating
             from a given user.
    """
    def wrapper(i_similarity):
        i_similarity = np.copy(i_similarity)
        for i in range(i_similarity.size):
            if not np.isnan(i_similarity[i]):
                if rating_matrix[user, i] == 0:
                    i_similarity[i] = np.NaN
        return neighborhood_f(i_similarity)
    return wrapper


#----------------- DIFFERENT NEIGHBORHOOD GENERATOR FUNCTIONS------------#


def item_neighborhood_generator(lines, rating_matrix, neighborhood_f):
    """
    Generate item neighborhood functions for each prediction to be made with the given neighborhood function.
    :param lines: numpy array containing predictions to be computed
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items
    :param neighborhood_f: neighborhood function
    :return: list of item neighborhood functions
    """
    neighborhood_fs = []
    for _, item, _ in lines:
        item = int(item) - 1
        neighborhood_fs.append(item_neighborhood(rating_matrix, item, neighborhood_f))
    return np.asarray(neighborhood_fs)


def user_neighborhood_generator(lines, rating_matrix, neighborhood_f):
    """
    Generate user neighborhood functions for each prediction to be made with the given neighborhood function.
    :param lines: numpy array containing predictions to be computed
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items
    :param neighborhood_f: neighborhood function
    :return: list of user neighborhood functions
        """
    neighborhood_fs = []
    for user, _, _ in lines:
        user = int(user) - 1
        neighborhood_fs.append(user_neighborhood(rating_matrix, user, neighborhood_f))
    return np.asarray(neighborhood_fs)


# --------------------------PREDICTION FUNCTIONS-------------------------#

def user_based_prediction(u_mean, neigh_mean, neigh_rating, neigh_similarity, w=None):
    """
    Compute the user based rating prediction. Function based on Equation (2.4) from "Recommender
    Systems - The textbook, Charu C. Aggarwal"
    :param u_mean: mean rating for a user
    :param neigh_mean: numpy array representing the neighbors rating means
    :param neigh_rating: numpy array representing the neighbors ratings for a given item
    :param neigh_similarity: numpy array representing the neighbors similarity
    :param w: numpy weight vector. If None, no weight is used during the computation of the rating prediction.
    :return: user based rating prediction
    """
    if len(neigh_mean) == 0:
        return np.NaN
    else:
        if w is None:
            w = np.ones(neigh_mean.size)
        neigh_similarity = w * neigh_similarity
        neigh_rating = neigh_rating - neigh_mean
        pred = u_mean + ((neigh_similarity @ neigh_rating) / np.abs(neigh_similarity).sum())
        return pred


def user_based_prediction_rounded(user_based_prediction_f):
    """
    Compute the user based rating prediction with a given user based prediction function rounding it up to nearest 0.5.
    :param user_based_prediction_f: user based prediction function
    :return: function that takes as input a mean rating for a user, neighbor rating means, neighbor ratings for a
             given item, neighbor similarity and weight vector and computes the rating prediction with a given user
             based prediction function rounding it up to nearest 0.5.
    """
    def wrapper(u_mean, neigh_mean, neigh_rating, neigh_similarity, w=None):
        pred = user_based_prediction_f(u_mean, neigh_mean, neigh_rating, neigh_similarity, w)
        if not np.isnan(pred):
            pred = min([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], key=lambda x: abs(x - pred))
        return pred
    return wrapper


def user_based_user_item_prediction(u, i, rating_matrix, u_similarity, neighborhood_f, prediction_f):
    """
    Compute the user based rating prediction of a user for a given item with a given user based prediction function.
    The neighbors of the given user are computed using the given neighborhood function.
    :param u: user
    :param i: item
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items
    :param u_similarity: numpy array representing the similarity of a user with respect to all the other users
    :param neighborhood_f: neighborhood function
    :param prediction_f: user based prediction function
    :return: user based rating prediction
    """
    u_rating = rating_matrix[u]
    u_mean = mean_rating(u_rating)
    neigh = neighborhood_f(u_similarity)
    if len(neigh) == 0:
        return prediction_f(u_mean, np.asarray([]), np.asarray([]), np.asarray([]))
    else:
        neigh_similarity = u_similarity[neigh]
        neigh_rating = rating_matrix[neigh, i]
        neigh_mean = np.apply_along_axis(mean_rating, 1, rating_matrix[neigh])
        return prediction_f(u_mean, neigh_mean, neigh_rating, neigh_similarity)


def item_based_prediction(u_neigh_rating, neigh_similarity, w=None):
    """
    Compute the item based rating prediction. Function based on Equation (2.15) from "Recommender
    Systems - The textbook, Charu C. Aggarwal"
    :param u_neigh_rating: numpy array representing the ratings of the user for neighbor items
    :param neigh_similarity: numpy array representing the similarity of the neighbor items
    :param w: numpy weight vector. If None, no weight is used during the computation of the rating prediction
    :return: rating prediction
    """
    if len(u_neigh_rating) == 0:
        return np.NaN
    else:
        if w is None:
            w = np.ones(neigh_similarity.size)
        neigh_similarity = w * neigh_similarity
        return (u_neigh_rating @ neigh_similarity) / np.abs(neigh_similarity).sum()


def item_based_prediction_rounded(item_based_prediction_f):
    """
    Compute the item based rating prediction with a given item based prediction function rounding it up to nearest 0.5.
    :param item_based_prediction_f: item based prediction function
    :return: function that takes as input the ratings of the user for neighbor items, the similarity of neighbor items
             and a weight vector to compute the item based rating prediction with a given item based prediction function
             rounding it up to nearest 0.5.
    """
    def wrapper(u_neigh_rating, neigh_similarity, w=None):
        pred = item_based_prediction_f(u_neigh_rating, neigh_similarity, w)
        if not np.isnan(pred):
            pred = min([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], key=lambda x: abs(x - pred))
        return pred
    return wrapper


def item_based_user_item_prediction(u, rating_matrix, i_similarity, neighborhood_f, prediction_f):
    """
    Compute the item based rating prediction of a user for a item with a given item based prediction function.
    The neighbors of the item are computed using the given neighborhood function.
    :param u: user
    :param rating_matrix: (n x i) rating matrix, where n is the number of users and i the number of items
    :param i_similarity: numpy array representing the similarity of a item with respect to all the other items
    :param neighborhood_f: neighborhood function
    :param prediction_f: item based prediction function
    :return: item based rating prediction
    """
    neigh = neighborhood_f(i_similarity)
    if len(neigh) == 0:
        return prediction_f(np.asarray([]), np.asarray([]))
    else:
        neigh_similarity = i_similarity[neigh]
        u_neigh_rating = rating_matrix[u, neigh]
        return prediction_f(u_neigh_rating, neigh_similarity)


# ---------------------------ERROR MEASURES------------------------------#

def mae(pred, real):
    """
    Mean Absolute Error between the given predictions and real data.
    :param pred: numpy array representing prediction values
    :param real: numpy array representing actual values
    :return: mean absolute error value
    """
    return np.abs(pred - real).mean()


def mse(pred, real):
    """
    Mean Squared Error between the given predictions and real data.
    :param pred: numpy array representing prediction values
    :param real: numpy array representing actual values
    :return: mean squared error value
    """
    return ((pred - real) @ (pred - real)) / len(pred)


def rmse(pred, real):
    """
    Root Mean Squared Error between the given predictions and real data.
    :param pred: numpy array representing prediction values
    :param real: numpy array representing actual values
    :return: root mean squared error value
    """
    return np.sqrt(mse(pred, real))


# --------------------------PREDICTION EVALUATION-----------------------#

def evaluate_predictions(pred_file, real_file, eval_f):
    """
    Compute the error measure between rating predictions and real ratings with the given error measure function.
    :param pred_file: path to a csv file containing rating predictions
    :param real_file: path to a csv file containing actual ratings
    :param eval_f: evaluation function, a.k.a error measure function
    :return: error measure
    """
    pred = load_data(pred_file)[:, 2].astype(np.float64)
    real = load_data(real_file)[:, 2].astype(np.float64)
    return eval_f(pred, real)

# ----------------------------RUN MODEL----------------------------------#

def run_model(train_data_file, test_data_file, save_file, rat_matrix_, similarity_matrix_, prediction_matrix_, neighborhood_f,
              prediction_f):
    """
    Train a recommender system whose structure is defined through string arguments. Make predictions with the trained
    recommender model and save them.
    :param train_data_file: path to the training data
    :param test_data_file: path to the test data
    :param save_file: saving file path
    :param rat_matrix_: rating matrix command
    :param similarity_matrix_: similarity matrix command
    :param prediction_matrix_: prediction matrix command
    :param neighborhood_f: neighborhood function command
    :param prediction_f: prediction function command
    :return: number of nan_predictions
    """
    train_data = load_data(train_data_file)
    test_data = load_data(test_data_file)
    rating_matrix_ = rating_matrix(train_data)
    rating_matrix_ = eval(rat_matrix_)
    timestamp_matrix_ = timestamp_matrix(train_data)
    similarity_matrix = eval(similarity_matrix_)
    prediction_matrix = eval(prediction_matrix_)
    neighborhood_f = eval(neighborhood_f)
    prediction_f = eval(prediction_f)
    prediction_matrix, nan_predictions = prediction_matrix(test_data, rating_matrix_, similarity_matrix, neighborhood_f, prediction_f)
    save_prediction_matrix(save_file, prediction_matrix)
    return nan_predictions


#------------------------------------------------------------------------#


if __name__ == '__main__':
    """
    REQUIRED COMMAND LINE ARGUMENTS
    train_data: path to the training data
    test_data: path to the test data
    save: saving file path
    rating_matrix_: Either "rating_matrix_" or "mean_center_rating_matrix(rating_matrix_)". The former computes the
                    rating matrix normally while the latter mean centers it as well.
    similarity_matrix_: Either "user_similarity_matrix(rating_matrix_, {similarity_f})" or 
                        "item_similarity_matrix(rating_matrix_, {similarity_f})". The former to be used for building a
                        user based recommender model while the latter for item based recommender systems.
                        {similarity_f} needs to be replaced with one of the similarity functions implemented in this
                        file.
    prediction_matrix_: Either "user_based_prediction_matrix" or "item_based_prediction_matrix". The former to be used
                        for building a user based recommender model while the latter for item based recommender systems.
    neighborhood_f: one of the neighborhood functions implemented in this file. Alternatively, the item neighborhood
                    generator or the user neighborhood generator can be used if needed.
                    "item_neighborhood_generator(test_data, rating_matrix_, {neighborhood_f}) or 
                    "user_neighborhood_generator(test_data, rating_matrix_, {neighborhood_f}) where {neighborhood_f}
                    needs to be replaced with a neighborhood function.
    prediction_f: "user_based_prediction" or "user_based_prediction_rounded(user_based_prediction)" for user based
                  recommender models. "item_based_prediction" or "item_based_prediction_rounded(item_based_prediction)"
                  for item based recommender models.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--rating_matrix", type=str, required=True)
    parser.add_argument("--similarity_matrix", type=str, required=True)
    parser.add_argument("--prediction_matrix", type=str, required=True)
    parser.add_argument("--neighborhood_f", type=str, required=True)
    parser.add_argument("--prediction_f", type=str, required=True)
    args = parser.parse_args()
    run_model(args.train_data, args.test_data, args.save, args.rating_matrix, args.similarity_matrix,
              args.prediction_matrix, args.neighborhood_f, args.prediction_f)
