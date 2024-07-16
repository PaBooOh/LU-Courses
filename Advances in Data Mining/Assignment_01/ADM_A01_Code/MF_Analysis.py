import joblib
import matplotlib.pyplot as plt
import numpy as np

def plot_loss():
    train_avg_results, test_avg_results = compute_avg_results()
    

    X = [i for i in range(len(train_avg_results['RMSE']))]

    fig = plt.figure(figsize=(12, 6))
    train_ax1 = fig.add_subplot(121)
    train_ax1.plot(X, train_avg_results['RMSE'], color='blue', label='RMSE')
    train_ax1.plot(X, train_avg_results['MAE'], color='orange', label='MAE')
    test_ax2 = fig.add_subplot(122)
    test_ax2.plot(X, test_avg_results['RMSE'], color='blue', label='RMSE')
    test_ax2.plot(X, test_avg_results['MAE'], color='orange', label='MAE')

    train_ax1.set_xlabel('Epoch')
    test_ax2.set_xlabel('Epoch')
    train_ax1.set_ylabel('Error')
    test_ax2.set_ylabel('Error')

    train_ax1.legend()
    test_ax2.legend()

    train_ax1.set_title('Performance on training set')
    test_ax2.set_title('Performance on test set')
    
    plt.show()


def compute_avg_results():
    train_avg_results = dict()
    test_avg_results = dict()

    train_rmse_tmp = np.zeros(75)
    train_mae_tmp = np.zeros(75)
    test_rmse_tmp = np.zeros(75)
    test_mae_tmp = np.zeros(75)
    for ID in range(1,6):
        train_results_path = 'Process-' + str(ID) + '_training_results'
        test_results_path = 'Process-' + str(ID)+ '_testing_results'
        train_results = np.array(joblib.load(train_results_path))
        test_results = np.array(joblib.load(test_results_path))

        train_rmse_tmp += train_results[:,0]
        train_mae_tmp += train_results[:,1]
        test_rmse_tmp += test_results[:,0]
        test_mae_tmp += test_results[:,1]
    train_avg_results['RMSE'] = train_rmse_tmp/5
    train_avg_results['MAE'] = train_mae_tmp/5
    test_avg_results['RMSE'] = test_rmse_tmp/5
    test_avg_results['MAE'] = test_mae_tmp/5

    return train_avg_results, test_avg_results

def approximate():
    U_path = 'U_vecProcess-5'
    M_path = 'M_vecProcess-5'
    U = joblib.load(U_path)
    M = joblib.load(M_path)
    pred = np.dot(U, M)
    for index, element in np.ndenumerate(pred):
        i, j = index
        if element > 5:
            pred[i][j] = 5
            continue
        elif element < 1:
            pred[i][j] = 1
            continue
        pred[i][j] = round(element) 
    print(pred)


if __name__ == "__main__":

    # 1) plot results
    plot_loss()

    # 2) approximate
    approximate()
