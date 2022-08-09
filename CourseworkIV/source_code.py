import numpy as np
import argparse
import os


#---------------------------THEORETICAL EXPLANATION----------------------#
"""
Matrix factorisation, a collaborative filtering solution for recommendation systems, is implemented in this Python
script. Before explaining Matrix factorisation in detail, we introduce the definition of recommender task and briefly
explain the collaborative filtering technique.

--RECOMMENDATION TASK
    Given a set of users U and a set of items I, we want to learn a function that predicts, based on past data, the
    rating of a given user u for a given item i.
    
--COLLABORATIVE FILTERING
    Collaborative filtering leverages the "wisdom of the crowd" to perform recommendation tasks. Indeed, the basic
    assumption underlying the collaborative filtering approach lies in the fact that customers who have agreed in the
    past, will agree in the future.
    
--MATRIX FACTORISATION
    Matrix factorisation is an embedding collaborative filtering model. Given a (u x i) rating matrix R, where u is the
    number of users and i is the number of items, the task of the model is to learn two embedding matrices U and I,
    such that the product of the latter matrices is a good approximation of the rating matrix R. U is called the user
    embedding matrix and its size is (u x f), where u is the number of users and f is the number of factors. Similarly,
    I is the item embedding matrix, and its size is (f x i), where i is the number of items and f is the number of
    factors.

    The rows of the user embedding matrix represent the features of the users, also known as factors or embeddings.
    Correspondingly, the columns of the item embedding matrix are the items' features.

    As an example, features for a movie might be its genre, plot, director and so on. However, matrix factorisation
    learns these features in an automated fashion; thus, no information about the users and items needs to be known
    a priori when using the matrix factorisation technique.

    Once Matrix factorisation learns the embedding matrices U and I, the computation of a new prediction of a given
    user u for a given item i is based on the dot-product between the u's factor vector and the i's factor vector.
    
    --LEARNING ALGORITHM
        The main idea behind finding matrices U and I such that their product is a good approximation of the rating
        matrix R is based on the minimisation of an objective function. In other words, we need to minimise the error
        between the predicted ratings and the actual ratings across the training set. A commonly adopted loss function
        is the mean squared error, mainly due to its simplicity. However, other objective functions have proved to give
        better results.

        A common algorithm used to minimise the objective function is stochastic gradient descent. It consists of
        looping over the whole training dataset, one instance at a time, and adjusting the user and item factor vectors
        using the prediction error according to a given learning rate and regularisation term. The learning rate is a
        tuning parameter that determines the step size while moving toward a minimum of a given loss function.
        In contrast, the regularisation term is a tuning parameter that is commonly used to prevent the risk of
        overfitting. Indeed, the higher the regularisation term, the more the factors are shrunk towards zero,
        preventing the training process from learning complex models.

--PSEUDOCODE
    This implementation of the Matrix factorisation technique is based on the pseudocode provided by
    Dr. Stuart E. Middleton in the Modern Recommender Systems slides. In addition, the code implemented here includes
    the learning of user and item biases, as it has been realised early on that recommender systems work a lot better
    when accounting for them. The intuition of why this is the case can be grasped through an example of movie rating
    predictions. 
    
    In movie rating scenarios, different users will have a different tendency to give better or worse
    ratings than the average. If we keep track of that bias for each user, we can just try to predict the difference
    from that bias, making the task of a recommender system much more straightforward. Also, the same thing needs to be
    done for the movies. In this case, the bias of a movie could be described as how well this movie is rated
    compared to the average across all movies. The whole point of this is to remove the bias given by users or
    movies. As an example, suppose that Anna tends to rate all movies 2 stas too high; thus, in order to compare
    her ratings to other users, you must remove 2 stars to all of her ratings. Equivalently, if you want to predict the
    rating Anna will give to a movie, you have to add 2 stars to the score you would get if Alice was an "average" user.

    Given the inclusion of the bias and item vectors, the computation of a new prediction of a given user u for a given
    item i is based on the dot-product between the u's factor vector and the i's factor vector, which needs to be
    added to the sum of the u's bias and i's bias, i.e. u_factor_vector @ i_factor_vector + u_bias + i_bias.

    The pseudocode of my implementation is proposed below.

    Training input = ratings(N_user, N_item) with empty values removed
    N_factor = ?  - to be selected by the user 
    Learning rate γ = ?  - to be selected by the user
    Regularization λ = ?  -to be selected by the user
    Randomly initialise item matrix q(N_item, N_factor) 
    Randomly initialise user matrix p(N_user, N_factor)
    if willing to include the user and item bias vectors in the training task
        Randomly initialise user bias vector r(N_user)
        Randomly initialise item bias vector s(N_items)
    loop for N_iterations
        make a randomly shuffled set of (u,i) pairs
            loop on each (u,i) pair 
                if using user and item bias vectors 
                    predicted_rating = q_i . p_u + r_u + s_i
                else
                    predicted_rating = q_i . p_u
                actual_rating = ratings(u,i) 
                error = actual_rating - predicted_rating
                p(u,*) += γ * ( error * q(i, *) - λ * p(u,*) ) -adjust the given user factor vector so that to reduce 
                                                                 the loss
                q(i,*) += γ * ( error * p(u, *) - λ * q(i,*) ) -adjust the given item factor vector so that to reduce
                                                                the loss
                if using user and item bias vectors
                r_u += γ * (error - λ * r_u) -adjust the given user bias so that to reduce the loss
                s_i += γ * (error - λ * s_i) -adjust the given item bias so that to reduce the loss
"""

# --------------------------IO NUMPY ARRAYS------------------------------#


def load_file(file):
    """
    Load a numpy file.
    :param file: file path
    :return: object representation of the numpy file
    """
    return np.load(file, allow_pickle=True)


def load_training_data(file):
    """
    Load training data from a numpy file.
    :param file: file path
    :return: numpy matrix
    """
    return load_file(file)[:, :3]


def load_predicting_data(file):
    """
    Load data for which rating predictions need to be computed.
    :param file: file path
    :return: numpy matrix
    """
    return load_file(file)


def save_file(file, x):
    """
    Save numpy object into a file.
    :param file: file path
    :param x: numpy object
    :return: None
    """
    np.save(file, x, allow_pickle=True)

#-----------------------------OS FUNCTIONS------------------------------#


def createdir(path):
    """
    Create directory. If the directory already exists, it will not be overwritten.
    :param path: directory path
    :return: None
    """
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created")
    except FileExistsError:
        print(f"Directory '{path}' already exists")


def write_eval(dir, e, train_eval, valid_eval):
    """
    Create a text file called "evals.txt" inside a directory and write the prediction errors in the training
    and validation dataset obtained in a given epoch. If the text file already exists, information about the results
    achieved in the epoch are appended at the end of the file.
    :param dir: directory path
    :param e: epoch number
    :param train_eval: prediction error in the training dataset
    :param valid_eval: prediction error in the validation dataset
    :return: None
    """
    with open(os.path.join(dir, "evals.txt"), "a") as f:
        f.write(str((train_eval, valid_eval, e)) + "\n")


# -----------------------------MATRIX-VECTOR FUNCTIONS---------------------------------#


def user_matrix(n_factors, n_users):
    """
    Build a (f x u) user embedding matrix, where f is the number of factors and u is the number of users.
    Every entry in the matrix is drawn from a normal gaussian distribution.
    :param n_factors: number of factors
    :param n_users: number of users
    :return: (f x u) user embedding matrix, where f is the number of factors and u is the number of users.
    """
    return np.random.randn(n_factors, n_users)


def item_matrix(n_factors, n_items):
    """
    Build a (f x i) item embedding matrix, where f is the number of factors and i is the number of items.
    Every entry in the matrix is drawn from a normal gaussian distribution.
    :param n_factors: number of factors
    :param n_items: number of items
    :return: (f x i) item embedding matrix, where f is the number of factors and i is the number of items.
    """
    return np.random.randn(n_factors, n_items)


def user_bias_vector(n_users):
    """
    Build u-dimensional user bias vector, where u is the number of users.
    Every entry in the vector is drawn from a normal gaussian distribution.
    :param n_users: number of users
    :return: u-dimensional user bias vector
    """
    return np.random.randn(n_users)


def item_bias_vector(n_items):
    """
    Build i-dimensional item bias vector, where i is the number of items.
    Every entry in the vector is drawn from a normal gaussian distribution.
    :param n_items: number of items
    :return: i-dimensional item bias vector
    """
    return np.random.randn(n_items)


#-----------------------------ADJUSTMENT FUNCTIONS-------------------------------#


def adjust_factor(gam, lam):
    """
    Modify user/item factor vectors adjusting by the prediction error according to a given learning rate and
    regularisation term.
    :param gam: learning rate
    :param lam: regularisation term
    :return: a function that modifies user/item factor vectors adjusting by the prediction error according to a given
    learning rate and regularisation term.
    """
    def wrapper(error, factors1, factors2):
        """
        Adjust factor vectors using the formula factors1 + gam * (error * factors2 - lam * factors1), where factors1 are
        the factor vectors to be adjusted, error is the vector of prediction errors and factors2 are the supporting
        factor vectors.
        :param error: vector of prediction errors
        :param factors1: factor vectors to be adjusted
        :param factors2: supporting factor vectors
        :return: adjusted factor vectors
        """
        factors1 = factors1.squeeze()
        factors2 = factors2.squeeze()
        factors1_adjust = factors1 + gam * (error * factors2 - lam * factors1)
        if factors1_adjust.ndim == 1:
            factors1_adjust = np.expand_dims(factors1_adjust, 1)
        return factors1_adjust
    return wrapper


def adjust_bias(gam, lam):
    """
    Modify user/item bias vectors adjusting by the prediction error according to a given learning rate and
    regularisation term.
    :param gam: learning rate
    :param lam: regularisation term
    :return: a function that modifies user/item bias vectors adjusting by the prediction error according to a given
    learning rate and regularisation term.
    """
    def wrapper(error, bias):
        """
        Adjust bias vectors using the formula bias + gam * (error - lam * bias), where bias is
        the bias vectors to be adjusted and error is the vector of prediction errors.
        :param error: vector of prediction errors
        :param bias: bias vectors
        :return: adjusted bias vectors
        """
        bias_adjust = bias + gam * (error - lam * bias)
        return bias_adjust
    return wrapper


#----------------------------PREDICTION FUNCTIONS--------------------------------#


def pred(u_factors, i_factors, u_biases=None, i_biases=None):
    """
    Compute rating prediction/s of a user or a set of users for a given item or set of items.
    The computation of the prediction/s changes according to whether the parameters u_biases and i_biases are "None"
    or not. If they are "None", a single prediction is computed with the expression u_factors @ i_factors. Otherwise,
    it is computed with the expression (u_factors @ i_factors) + u_biases + i_biases. Notice that in the case matrices
    are provided as arguments to the parameters u_factors and i_factors, the rating predictions are computed in a
    point-wise fashion.
    :param u_factors: f-dimensional user factors vector or (f x u) users factors matrix, where f is the number of
                      factors and u is the number of users whose rating predictions need to be computed
    :param i_factors: f-dimensional item factors vector or (f x i) items factors matrix, where f is the number of
                      factors and i is the number of items for which rating predictions need to be computed
    :param u_biases: scalar user bias or u-dimensional scalar users bias vector, where u is the number of users whose
                     rating predictions need to be computed
    :param i_biases: scalar item bias or i-dimensional scalar items bias vector, where i is the number of items for
                     which rating predictions need to be computed
    :return: scalar rating prediction or a vector of rating predictions
    """
    u_factors = u_factors.squeeze()
    i_factors = i_factors.squeeze()
    if u_factors.ndim == 1:  # if the provided inputs are scalar
        pred = u_factors @ i_factors
        if u_biases is not None:
            pred = pred + u_biases + i_biases
        return pred
    else:  # otherwise, if the provided inputs are matrices
        pred = np.sum(u_factors * i_factors, axis=0)
        if u_biases is not None:
            pred = pred + u_biases + i_biases
        return pred


def pred_dataset(dataset, u_matrix, i_matrix, u_bias_vector=None, i_bias_vector=None):
    """
    Compute the rating predictions for a dataset given a user embedding matrix, an item embedding matrix,
    and eventual user and item bias vectors. For more information regarding the modality of the rating predictions
    computation, consult the documentation of function "pred".
    :param dataset: dataset containing the (user,item) pairs for which rating predictions need to be computed.
    :param u_matrix: (f x u) user embedding matrix, where f is the number of factors and u is the number of users.
    :param i_matrix: (f x i) item embedding matrix, where f is the number of factors and i is the number of items.
    :param u_bias_vector: u-dimensional user bias vector, where u is the number of users.
    :param i_bias_vector: i-dimensional item bias vector, where i is the number of items.
    :return: 7-tuple
                1st element -> vector of rating predictions
                2nd element -> vector of user ids whose rating predictions have been computed
                3rd element -> vector of item ids for which rating predictions have been computed
                4th element -> factors matrix of the users whose rating predictions have been computed
                5th element -> factors matrix of the items for which rating predictions have been computed
                6th element -> vector of users' biases whose rating predictions have been computed.
                7th element -> vector of items' biases for which rating predictions have been computed
    """
    users = dataset[:, 0].astype(int) - 1
    items = dataset[:, 1].astype(int) - 1
    u_factors = u_matrix[:, users]
    i_factors = i_matrix[:, items]
    if u_bias_vector is not None:
        u_biases = u_bias_vector[users]
        i_biases = i_bias_vector[items]
        return pred(u_factors, i_factors, u_biases, i_biases), users, items, u_factors, i_factors, u_biases, i_biases
    else:
        return pred(u_factors, i_factors), users, items, u_factors, i_factors, None, None


# ---------------------------ERROR MEASURES------------------------------#


def mae(pred, real):
    """
    Mean Absolute Error between the given predictions and real data.
    :param pred: numpy array representing prediction values
    :param real: numpy array representing actual values
    :return: mean absolute error value
    """
    x = np.abs(pred - real).mean()
    return x


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


#-----------------------------TRAINING-VALIDATION SPLIT FUNCTIONS---------------------------#


def train_validation_split(dataset, train_rate=0.8):
    """
    Split a dataset into random train and validation subsets.
    :param dataset: dataset that needs to be split
    :param train_rate: the proportion of the dataset to include in the train split. The remaining proportion is included
                       in the validation split
    :return: a pair containing the train-validation split
    """
    data_len = len(dataset)
    train_end_index = int(data_len * train_rate)
    np.random.shuffle(dataset)
    return dataset[:train_end_index], dataset[train_end_index:]


def train_validation_split_files(dataset_path, train_data_path, valid_data_path, train_rate=0.8):
    """
    Load a dataset from a file and split it into random train and validation subsets. The train and validation splits
    are then saved into a file.
    :param dataset_path: path to the dataset that needs to be split
    :param train_data_path: path where to save the train split
    :param valid_data_path: path where to save the validation split
    :param train_rate: the proportion of the dataset to include in the train split. The remaining proportion is included
                       in the validation split
    :return: None
    """
    dataset = load_training_data(dataset_path)
    train_data, valid_data = train_validation_split(dataset, train_rate)
    save_file(train_data_path, train_data)
    save_file(valid_data_path, valid_data)


#-----------------------------TRAINING ITERATIVE FUNCTIONS----------------------------#


def train_iters(train_data, u_matrix, i_matrix, u_bias_vector, i_bias_vector, adjust_user_factors, adjust_item_factors,
                adjust_user_bias, adjust_item_bias, batch_size, eval_f):
    """
    Loop over the whole training dataset and adjust on each batch the user embedding matrix, item embedding matrix and
    eventual user and item biases with the given adjusting functions. The training dataset is shuffled before iterating
    over it. The average predictive error, computed with the given evaluation function, is returned as output.
    :param train_data: training dataset
    :param u_matrix: (f x u) user embedding matrix, where f is the number of factors and u is the number of users.
    :param i_matrix: (f x i) item embedding matrix, where f is the number of factors and i is the number of items.
    :param u_bias_vector: u-dimensional user bias vector, where u is the number of users.
    :param i_bias_vector: i-dimensional item bias vector, where i is the number of items.
    :param adjust_user_factors: function adjusting user factor vectors
    :param adjust_item_factors: function adjusting item factor vectors
    :param adjust_user_bias: function adjusting user bias
    :param adjust_item_bias: function adjusting item bias
    :param batch_size: batch size
    :param eval_f: evaluation function
    :return: average prediction error on the training dataset
    """
    train_evals = []
    indexes = np.linspace(0, len(train_data) - 1, len(train_data) - 1, dtype=int)
    np.random.shuffle(indexes)
    s_index = 0
    end_index = batch_size
    while s_index < len(train_data) - 1:
        batch = train_data[indexes[s_index : end_index]]
        # Evaluate the rating prediction performance on the training dataset
        train_eval, real_rating, pred_rating, users, items, u_factors, i_factors, u_biases, i_biases = evals(batch,
                                                            u_matrix, i_matrix, eval_f, u_bias_vector, i_bias_vector)
        # Compute prediction error
        error = real_rating - pred_rating
        # Adjust user factor vectors
        u_factors_adjusts = adjust_user_factors(error, u_factors, i_factors)
        # Adjust item factor vectors
        i_factors_adjusts = adjust_user_factors(error, i_factors, u_factors)
        # Save the adjusted factor vectors back into their corresponding matrices
        u_matrix[:, users] = u_factors_adjusts
        i_matrix[:, items] = i_factors_adjusts
        # If user and item biases are being used in the rating prediction computations
        if u_bias_vector is not None:
            # Adjust user bias
            u_bias_adjust = adjust_user_bias(error, u_biases)
            # Adjust item bias
            i_bias_adjust = adjust_item_bias(error, i_biases)
            # Save the adjusted user and item biases back into their corresponding vectors
            u_bias_vector[users] = u_bias_adjust
            i_bias_vector[items] = i_bias_adjust

        train_evals.append(train_eval)
        s_index += batch_size
        end_index += batch_size
        print(f"Batch eval:{train_eval}")

    return np.asarray(train_evals).mean()


#-------------------------------EVALUATION FUNCTIONS------------------------------#


def evals(data, u_matrix, i_matrix, eval_f, u_bias_vector=None, i_bias_vector=None):
    """
    Evaluate the rating prediction performance on a dataset using a given evaluation function.
    :param data: dataset on which to evaluate rating prediction performance
    :param u_matrix: (f x u) user embedding matrix, where f is the number of factors and u is the number of users.
    :param i_matrix: (f x i) item embedding matrix, where f is the number of factors and i is the number of items.
    :param eval_f: evaluation function
    :param u_bias_vector: u-dimensional user bias vector, where u is the number of users.
    :param i_bias_vector: i-dimensional item bias vector, where i is the number of items.
    :return: 9-tuple
                1st element -> scalar error measure
                2nd element -> vector of real ratings
                3rd element -> vector of predicted ratings
                4th element -> vector of user ids whose rating predictions have been computed
                5rd element -> vector of item ids for which rating predictions have been computed
                6th element -> factors matrix of the users whose rating predictions have been computed
                7th element -> factors matrix of the items for which rating predictions have been computed
                8th element -> vector of users' biases whose rating predictions have been computed.
                9th element -> vector of items' biases for which rating predictions have been computed
    """
    real_rating = data[:, 2].astype(np.float16)
    pred_rating, users, items, u_factors, i_factors, u_biases, i_biases = pred_dataset(data, u_matrix, i_matrix,
                                                                                       u_bias_vector, i_bias_vector)
    return eval_f(pred_rating, real_rating), real_rating, pred_rating, users, items, u_factors, i_factors, u_biases, i_biases


#----------------------------TRAINING FUNCTIONS-----------------------------------#


def epoch_io(e, u_matrix, i_matrix, u_bias_vector, i_bias_vector, train_eval, test_eval, train_evals, test_evals, save_path):
    """
    Create directory for a given epoch and save its statistics inside. The statistics that are saved for a given epoch
    are the user and item embedding matrix, the user and item bias vectors if they are being used in the rating
    prediction calculations and the epoch's prediction error in the training and test datasets together with the results
    of the previous epochs.
    :param e: current epoch number
    :param u_matrix: (f x u) user embedding matrix, where f is the number of factors and u is the number of users.
    :param i_matrix: (f x i) item embedding matrix, where f is the number of factors and i is the number of items.
    :param u_bias_vector: u-dimensional user bias vector, where u is the number of users.
    :param i_bias_vector: i-dimensional item bias vector, where i is the number of items.
    :param train_eval: prediction error in the training dataset of current epoch
    :param test_eval: prediction error in the test dataset of current epoch
    :param train_evals: vector of each epoch's prediction error in the training dataset
    :param test_evals: vector of each epoch's prediction error in the test dataset
    :param save_path: path where to create epoch directory and save epoch statistics inside
    :return: None
    """
    dir = os.path.join(save_path, f"epoch{e}")
    createdir(dir)
    save_file(os.path.join(dir, "u_matrix"), u_matrix)
    save_file(os.path.join(dir, "i_matrix"), i_matrix)
    if u_bias_vector is not None:
        save_file(os.path.join(dir, "u_bias_vector"), u_bias_vector)
        save_file(os.path.join(dir, "i_bias_vector"), i_bias_vector)
    save_file(os.path.join(dir, "train_evals"), np.asarray(train_evals))
    save_file(os.path.join(dir, "test_evals"), np.asarray(test_evals))
    write_eval(save_path, e, train_eval, test_eval)
    print(f"Epoch eval:{(train_eval, test_eval)}")


def train(train_data, val_data, u_matrix, i_matrix, u_bias_vector, i_bias_vector, adjust_user_factors,
          adjust_item_factors, adjust_user_bias, adjust_item_bias, batch_size, start_epoch, end_epoch, eval_f, save_path):
    """
    Train a recommender system starting from the given user embedding matrix, item embedding matrix and eventual user
    and item bias vectors. The recommender system is trained on the given training dataset and its rating prediction
    performance is evaluated with the given evaluation function on the validation dataset. During the training process,
    the error between the predicted and true rating is minimised using stochastic gradient descent and the given
    adjusting functions. The user can specify the batch size and the maximum number of epochs for the training
    procedure. During the training process, several statistics are saved for each epoch. This allows finer-grained
    analysis of the predictive performance and enables recovering training without any loss of information in case of
    accidental crashes.
    :param train_data: training dataset
    :param val_data: validation dataset
    :param u_matrix: (f x u) user embedding matrix, where f is the number of factors and u is the number of users.
    :param i_matrix: (f x i) item embedding matrix, where f is the number of factors and i is the number of items.
    :param u_bias_vector: u-dimensional user bias vector, where u is the number of users.
    :param i_bias_vector: i-dimensional item bias vector, where i is the number of items.
    :param adjust_user_factors: function adjusting user factor vectors
    :param adjust_item_factors: function adjusting item factor vectors
    :param adjust_user_bias: function adjusting user bias
    :param adjust_item_bias: function adjusting item bias
    :param batch_size: batch size
    :param start_epoch: start epoch number
    :param end_epoch: end epoch number
    :param eval_f: evaluation function
    :param save_path: path where to save statistics of the whole training
    :return: None
    """
    train_evals = []
    test_evals = []
    # For every epoch
    for e in range(start_epoch, end_epoch):
        # Loop over the whole training dataset and adjust user/item factors and eventual user/item biases on every batch
        train_eval = train_iters(train_data, u_matrix, i_matrix, u_bias_vector, i_bias_vector, adjust_user_factors,
                                 adjust_item_factors, adjust_user_bias, adjust_item_bias, batch_size, eval_f)
        # Evaluate rating predictive performance on the validation dataset
        test_eval, _, _, _, _, _, _, _, _ = evals(val_data, u_matrix, i_matrix, eval_f, u_bias_vector, i_bias_vector)
        train_evals.append(train_eval)
        test_evals.append(test_eval)
        # Save epoch statistics into files
        epoch_io(e, u_matrix, i_matrix, u_bias_vector, i_bias_vector, train_eval, test_eval, train_evals, test_evals,
                 save_path)


def run_training(train_data, test_data, n_factors, u_matrix, i_matrix, u_bias_vector, i_bias_vector, adjust_user_factors,
                 adjust_item_factors, adjust_user_bias, adjust_item_bias, batch_size, start_epoch, end_epoch, eval_f,
                 save_path):
    """
    Run training process with the given training arguments. For more information regarding the training process, the
    reader is referred to the function "train".
    :param train_data: path to training dataset file
    :param test_data: path to test dataset file
    :param n_factors: number of factors for the user and item embedding matrices
    :param u_matrix: if it is "None", a random user embedding matrix is created drawing samples from a normal gaussian
                     distribution. Otherwise, it is assumed to be a path to a user embedding matrix file,
                     which is then  loaded.
    :param i_matrix: if it is "None", a random item embedding matrix is created drawing samples from a normal gaussian
                     distribution. Otherwise, it is assumed to be a path to an item embedding matrix file,
                     which is then  loaded.
    :param u_bias_vector: If it is "None" and adjust_user_bias is not "None", a random user bias vector is created
                          drawing samples from a normal gaussian distribution. Otherwise, if it is not "None", it
                          is assumed to be a user bias vector file path, which is then loaded.
    :param i_bias_vector: If it is "None" and adjust_item_bias is not "None", a random item bias vector is created
                          drawing samples from a normal gaussian distribution. Otherwise, if it is not "None", it
                          is assumed to be an item bias vector file path, which is then loaded.
    :param adjust_user_factors: function adjusting user factor vectors
    :param adjust_item_factors: function adjusting item factor vectors
    :param adjust_user_bias: function adjusting user bias
    :param adjust_item_bias: function adjusting item bias
    :param batch_size: batch size
    :param start_epoch: start epoch
    :param end_epoch: end epoch
    :param eval_f: evaluation function
    :param save_path: path where to save statistics of the whole training
    :return: None
    """
    # Load training dataset
    train_data = load_training_data(train_data)
    # Load test dataset
    test_data = load_training_data(test_data)
    # Extract number of users and items from both training and test datasets
    n_users, n_items = np.concatenate((train_data, test_data))[:, :2].astype(int).max(axis=0)

    if u_matrix is None:
        # Create random user embedding matrix
        u_matrix = user_matrix(n_factors, n_users)
        # Create random item embedding matrix
        i_matrix = item_matrix(n_factors, n_items)
    else:
        # Load user embedding matrix
        u_matrix = load_file(u_matrix)
        # Load item embedding matrix
        i_matrix = load_file(i_matrix)

    if u_bias_vector is None and adjust_user_bias is not None:
        # Create random user bias vector
        u_bias_vector = user_bias_vector(n_users)
        # Create random item bias vector
        i_bias_vector = item_bias_vector(n_items)
    elif u_bias_vector is not None:
        # Load user bias vector
        u_bias_vector = load_file(u_bias_vector)
        # Load item bias vector
        i_bias_vector = load_file(i_bias_vector)

    dir = "recommendy3"
    createdir(dir)
    save_path = os.path.join(dir, save_path)
    createdir(save_path)
    # Train recommender system
    train(train_data, test_data, u_matrix, i_matrix, u_bias_vector, i_bias_vector, adjust_user_factors,
          adjust_item_factors, adjust_user_bias, adjust_item_bias, batch_size, start_epoch, end_epoch, eval_f, save_path)


if __name__ == '__main__':
    """
    REQUIRED COMMAND LINE ARGUMENTS
    In what follows, R denotes required argument while NR stands for non-required.
    train_data: R-(string argument) path to training dataset file
    test_data: R-(string argument) path to test dataset file
    save: R-(string argument) path where to save the training's statistics
    n_factors: NR-(integer argument) number of factors for the user and item embedding matrix
    user_matrix: NR-(string argument) path to user embedding matrix file
    item_matrix: NR-(string argument) path to item embedding matrix file
    user_bias_vector: NR-(string argument) path to user vector file
    item_bias_vector: NR-(string argument) path to item vector file
    start_epoch: R-(integer argument) start epoch
    end_epoch: R-(integer argument) end epoch
    adjust_user_factors: R-(string argument) adjusting user factors function. "adjust_factor({gam},{lam})", where {gam}
                         needs to be replaced with a learning rate value and {lam} with a regularisation term value.
    adjust_item_factors: R-(string argument) adjusting item factors function. "adjust_factor({gam},{lam})", where {gam}
                         needs to be replaced with a learning rate value and {lam} with a regularisation term value.
    adjust_user_bias: NR-(string argument) adjusting user bias function. "adjust_bias({gam},{lam})", where {gam} needs to
                      be replaced with a learning rate value and {lam} with a regularisation term value.
    adjust_item_bias: NR-(string argument) adjusting item bias function. "adjust_bias({gam},{lam})", where {gam} needs to
                      be replaced with a learning rate value and {lam} with a regularisation term value.
    eval_f: R-(string argument) evaluation function. "mae", "mse" or "rmse". 
    batch size: R-(integer argument) batch size
    """
    np.seterr(all='raise')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--n_factors", type=int, required=False)
    parser.add_argument("--user_matrix", type=str, required=False)
    parser.add_argument("--item_matrix", type=str, required=False)
    parser.add_argument("--user_bias_vector", type=str, required=False)
    parser.add_argument("--item_bias_vector", type=str, required=False)
    parser.add_argument("--start_epoch", type=int, required=True)
    parser.add_argument("--end_epoch", type=int, required=True)
    parser.add_argument("--adjust_user_factors", type=str, required=True)
    parser.add_argument("--adjust_item_factors", type=str, required=True)
    parser.add_argument("--adjust_user_bias", type=str, required=False)
    parser.add_argument("--adjust_item_bias", type=str, required=False)
    parser.add_argument("--eval_f", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()
    if args.adjust_user_bias is None:
        args.adjust_user_bias = "None"
        args.adjust_item_bias = "None"
    # Run training
    run_training(args.train_data, args.test_data, args.n_factors, args.user_matrix, args.item_matrix,
                 args.user_bias_vector, args.item_bias_vector, eval(args.adjust_user_factors),
                 eval(args.adjust_item_factors), eval(args.adjust_user_bias), eval(args.adjust_item_bias),
                 args.batch_size, args.start_epoch, args.end_epoch, eval(args.eval_f), args.save)
