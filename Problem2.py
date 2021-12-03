import numpy as np
from numpy import array, linspace
from scipy.integrate import solve_ivp
import pylab
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Generate data set for lorenz attractor
def func(t, r):
    x, y, z = r 
    fx = 10 * (y - x)
    fy = 28 * x - y - x * z
    fz = x * y - (8.0 / 3.0) * z
    return array([fx, fy, fz], float)


r0 = [0, 1, 0]
sol = solve_ivp(func, [0, 50], r0, t_eval=linspace(0, 50, 700))
sol1 = solve_ivp(func, [0, 50], r0, t_eval=linspace(0, 50, 800))



'''
# and plot it
fig = pylab.figure()
ax = pylab.axes(projection="3d")
ax.plot3D(sol.y[0,:], sol.y[1,:], sol.y[2,:], 'blue')
pylab.show()
'''


training_set = sol.y[0:1,:]/15
test_set = sol1.y[0:1,:]/15
def run():
    # First train the model
    w_12 = np.random.rand(20,200)                       # Weight matrix between layer 1 and 2 (20x200)
    w_23 = np.random.rand(200,1)                        # Weight matrix between layer 2 and 3 (200x1)
    e = np.ones((1,1))                                    # Error
    error = []
    a=1
    b=0.5
    for i in range(50):
        e_temp = 0
        lr = 0.1 - i*(0.1-0.00001)/49
        for t in range(21,700):   
            # Forward computation
            # Calculating output
            x_1 = training_set[:,t-21:t-1]                         # Input in layer 1 (1x20)
            v_2 = np.dot(x_1,w_12)                              # Induced local field in layer 2 (1x200)
            y_2 = a*np.tanh(b*v_2)                          # Output in layer 2 (1x200)
            v_3 = np.dot(y_2,w_23)                              # Induced local field in layer 3 (1x1)
            y_3 = a*np.tanh(b*v_3)                          # Output in layer 3 (1x1)        
            # Compute error signal    
            e = training_set[:,t-1:t] - y_3                                         # Error in layer 3 (1x3)
            e_temp = e_temp + e[0,0]*e[0,0]
            # Backward computation
            delta_3 = b/a * e * (a + y_3) * (a - y_3)                       # Delta in layer 3 (1x1)
            delta_2 = b/a * (a + y_2) * (a - y_2) * np.transpose((np.dot(w_23,np.transpose(delta_3))))  # Delta in layer 2 (1x200)
            # Adjust synaptic weights
            w_12 = w_12 + lr * np.dot(np.transpose(x_1),delta_2)    
            w_23 = w_23 + lr * np.dot(np.transpose(y_2),delta_3)
        error.append(e_temp/679)  
    x = []
    for i in range(50):
        x.append(i)
    plt.plot(x,error)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.show()
    # Now test the model
    l_output = []               # Lorenz output
    a_output = []               # Actual output
    x1 = []
    for t in range(21,800):   
        # Forward computation
        # Calculating output
        x_1 = test_set[:,t-21:t-1]                         # Input in layer 1 (1x20)
        v_2 = np.dot(x_1,w_12)                              # Induced local field in layer 2 (1x200)
        y_2 = a*np.tanh(b*v_2)                         # Output in layer 2 (1x200)
        v_3 = np.dot(y_2,w_23)                              # Induced local field in layer 3 (1x1)
        y_3 = a*np.tanh(b*v_3)                         # Output in layer 3 (1x1) 
        l_output.append(y_3[0,0])
        a_output.append(test_set[0,t-1])  
        x1.append(t-21)
    
    plt.plot(x1,l_output,label="Lorenz output")
    plt.plot(x1,a_output,label="Actual output")
    plt.xlabel("Time Step")
    plt.ylabel("Output")
    plt.legend()
    plt.show()

    

run()
