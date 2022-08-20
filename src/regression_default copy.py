import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
def generate_dataset(n):
    x = []
    y = []
    random_x1 = np.random.rand()
    random_x2 = np.random.rand()
    for i in range(n):
        x1 = i
        x2 = i/2 + np.random.rand()*n
        x.append([1, x1, x2])
        y.append(random_x1 * x1 + random_x2 * x2 + 1)
    return np.array(x), np.array(y)

def mse(coef, x, y):
    return np.mean((np.dot(x, coef) - y)**2)/2
 
def gradients(coef, x, y):
    return np.mean(x.transpose()*(np.dot(x, coef) - y), axis = 1)
 
def multilinear_regression(coef, x, y, lr, b1 = 0.9, b2 = 0.999, epsilon = 1e-8):
    prev_error = 0
    m_coef = np.zeros(coef.shape)
    v_coef = np.zeros(coef.shape)
    moment_m_coef = np.zeros(coef.shape)
    moment_v_coef = np.zeros(coef.shape)
    t = 0
 
    while True:
        error = mse(coef, x, y)
        if abs(error - prev_error) <= epsilon:
            break
        prev_error = error
        grad = gradients(coef, x, y)
        t += 1
        m_coef = b1 * m_coef + (1-b1)*grad
        v_coef = b2 * v_coef + (1-b2)*grad**2
        moment_m_coef = m_coef / (1-b1**t)
        moment_v_coef = v_coef / (1-b2**t)

        delta = ((lr / moment_v_coef**0.5 + 1e-8) *
                 (b1 * moment_m_coef + (1-b1)*grad/(1-b1**t)))
 
        coef = np.subtract(coef, delta)
    return coef

def main():
    x, y = generate_dataset(200)

    print(f"x.shape: {x.shape}")
    print(f"x: {x}")
    print(f"y.shape: {y.shape}")
    print(f"y: {y}")

    # plotting
    mpl.rcParams['legend.fontsize'] = 12
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    
    ax.scatter(x[:, 1], x[:, 2], y, label ='y', s = 5)
    ax.legend()
    ax.view_init(45, 0)
    
    plt.show()

###################################################


    coef = np.array([0, 0, 0])
    c = multilinear_regression(coef, x, y, 1e-1)

    print(f"c.shape: {c.shape}")
    print(f"c: {c}")


    # plotting
    fig = plt.figure()
    ax = fig.gca(projection ='3d')
    
    ax.scatter(x[:, 1], x[:, 2], y, label ='y',
                    s = 5, color ="dodgerblue")
    
    ax.scatter(x[:, 1], x[:, 2], c[0] + c[1]*x[:, 1] + c[2]*x[:, 2],
                        label ='regression', s = 5, color ="orange")
    
    ax.view_init(45, 0)
    ax.legend()
    plt.show()  


main()