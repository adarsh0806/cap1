import numpy as np
import matplotlib.pyplot as plt

# small function for creating random sinusoidal dataset for regression
def generate_regression_data(num_pts):
    # generate true function
    true_x = np.linspace(0,1,200)
    true_y = np.sin(2*np.pi*true_x)

    # generate data from true function
    data_x = np.random.rand(num_pts)
    data_y = np.sin(2*np.pi*data_x)
    noise = 0.2*np.random.randn(num_pts)
    data_y+=noise
    
    # reshape data and concatenate
    data_x.shape = (len(data_x),1)
    data_y.shape = (len(data_y),1)
    data = np.concatenate((data_x,data_y), axis = 1)
    
    # reshape boundary and concatenate
    true_x.shape = (len(true_x),1)
    true_y.shape = (len(true_y),1)
    true_func = np.concatenate((true_x,true_y),axis = 1)
    
    # plot data and true function
    plot_data(data,true_func)
    
    # return data and function
    return data,true_func

# function - plot data with underlying target function generated in the previous Python cell
def plot_data(data,true_func):
    data_x = data[:,0]
    data_y = data[:,1]
    true_x = true_func[:,0]
    true_y = true_func[:,1]
    
    # plot target
    plt.plot(true_x,true_y,'r--',linewidth = 2.5)

    # plot data 
    plt.scatter(data_x,data_y,facecolor = 'b',edgecolor = 'k',linewidth = 2.5)

    plt.xlim(min(data_x)-0.1,max(data_x)+0.1)
    plt.ylim(min(data_y)-0.3,max(data_y)+0.3)
    plt.yticks([],[])
    plt.axis('off')   
    
# function - plot training and testing sets
def plot_train_test(data_train, data_test):
    # plot data 
    fig = plt.figure(figsize = (4,4))
    plt.scatter(data_train[:,0],data_train[:,1],color = 'k',linewidth = 2,marker = 'o')
    plt.scatter(data_test[:,0],data_test[:,1],color = 'k',linewidth = 1,marker = 's',edgecolor = 'lime', s = 50)
    
    # clean up plot
    plt.yticks([],[])
    plt.xlim([-0.1,1.1])
    plt.ylim([min(min(data_train[:,1]),min(data_test[:,1])) - 0.1,max(max(data_train[:,1]),max(data_test[:,1])) + 0.1])
    plt.axis('off') 
    plt.title('training (circles) and testing (squares) sets')
    
# plot approximation
def plot_approx(clf,data):
    r = np.linspace(min(data[:,0]),max(data[:,0]),300)[:, np.newaxis]

    # use regressor to make predictions across the input domain
    z = clf.predict(r)

    # plot regressor
    plt.plot(r,z,linewidth = 3,color = 'b')
    data_y = data[:,1]
    plt.ylim(min(min(data_y)-0.1,min(z)-0.1),max(max(data_y)+0.1,max(z)+0.1)) 