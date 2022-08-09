import numpy as np

# --------------------------LOADING CSV FUNCTIONS------------------------------#

# Loads a csv file and converts it into a numpy array
def load_data(file):
    with open(file) as f:
        lines = f.readlines()
    return np.asarray([line.strip('\n').split(',') for line in lines])


# --------------------------SAVING MATRIX TO FILE FUNCTIONS---------------------#

# Saves a numpy array into a file
def save_matrix(file, x, delimiter, fmt):
    np.savetxt(file, x, delimiter=delimiter, fmt=fmt)


# Saves a numpy array into a csv file
def save_matrix_csv(file, x, fmt):
    save_matrix(file, x, ',', fmt)


# ---------------------------ERROR MEASURES------------------------------#
# Mean Absolute Error
def mae(pred, real):
    return np.abs(pred - real).mean()


# Mean Squared Error
def mse(pred, real):
    return ((pred - real) @ (pred - real)) / len(pred)


# Root Mean Squared Error
def rmse(pred, real):
    return np.sqrt(mse(pred, real))

# --------------------------EVALUATE PREDICTIONS-----------------------#


# Evaluates predictions against real
def evaluate_predictions(pred_file, real_file, eval_file):
    pred = load_data(pred_file)[:, 2].astype(np.float64)
    real = load_data(real_file)[:, 2].astype(np.float64)
    x = mse(pred, real)
    y = rmse(pred, real)
    z = mae(pred, real)
    eval_metrics = np.asarray([x, y, z]).reshape((1, 3))
    save_matrix_csv(eval_file, eval_metrics, '%.12f')


if __name__ == '__main__':
    evaluate_predictions('comp3208_micro_pred.csv', 'comp3208_micro_gold.csv', 'results.csv')
