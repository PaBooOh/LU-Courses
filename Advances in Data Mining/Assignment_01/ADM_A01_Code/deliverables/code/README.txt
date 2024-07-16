=============================================================================================
1. Naive Method

There are 5 different function for 5 different method. error function to calculate rmse and mae. Cross function is for 5-fold validtion. 
Running the functions one by one is ok.
=============================================================================================
2. UV decomosition

You can change 'd' to get different results for the short side of U and V. ii is for the epoch times. There are two different versions 
one track the latter track the rmse and mae for each epoch the former does not.
=============================================================================================
3. Matrix Factorization

For this algorithm, there are two files: Matrix_Factorization.py & MF_Analysis.py, which are used for gradient descent and for analysis on results.
    *Matrix_Factorization.py:
Class CrossValidation is defined to split data set into 5 folds and class MF is used to train weight matrices U and M  to approximate 
the ratings data processed by methods in class CrossValidation. We import package 'multiprocessing' to run 5 models in parallel.
Just run the program and wait a few seconds for the ratings.dat to be processed. Then training process begin. For each epoch/iteration, 
RMSE/MAE on the data set will be printed for you to observe the tranining process. Here we set epoch/iteration to 75 and after 75 epochs,
the results of performance and two parameters matrices (U&M) will be saved to local disk for analysis and prediction later.

    *MF_Analysis.py
Once the parameters matrice and results of performance is saved, in this file, we use package 'joblib' to read them and slightly process the results
by method 'compute_avg_results' to get the average of the five results (train&test). Then, you can use method 'plot_loss' to show the training 
process (RMSE/MAE) and use method 'approximate' to get the prediction matrix.
=============================================================================================

