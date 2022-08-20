import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
 
def generate_dataset(n):
    x = []
    y = []
    random_x1 = np.random.rand()
    random_x2 = np.random.rand()
    for i in range(n):
        x1 = 2*i + np.random.rand() * 5 + np.sin(i) * 5
        x2 = -i + np.random.rand() * 5 + np.sin(i) * 5
        x.append([x1, x2])
        y.append(1 + np.random.rand() * 5 + np.sin(i) * 5)
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

def predict(x, coef):
    return np.sum(x * coef, axis=1)

def rmse(p: np.ndarray, t: np.ndarray, visual: bool = True) -> float:
    """
    Compute root mean square error (RMSE) between prediction and target signal.
    Scale up smaller signal to enable metric.
    """
    if visual:
        fig = plt.figure()
        plt.plot(p, label="prediction")
        plt.plot(t, label="target")
        plt.legend()

    return np.sqrt(((p-t)**2).mean())


def main():
    x, y = generate_dataset(200)

    fig = plt.figure()
    plt.plot(x, label="x")
    plt.plot(y, label="y")
    plt.legend()

    print(f"x.shape: {x.shape}")
    print(f"x: {x}")
    print(f"y.shape: {y.shape}")
    print(f"y: {y}")

###################################################

    ################ sklearn prediction
    reg = LinearRegression()
    reg.fit(x, y)

    print(f"skcoef.shape: {reg.coef_.shape}")
    print(f"skcoef: {reg.coef_}")

    ################ default prediction

    coef = np.array([0, 0])
    c = multilinear_regression(coef, x, y, 1e-1)

    print(f"c.shape: {c.shape}")
    print(f"c: {c}")


    ################ rafael prediction

    p = predict(x, c)
    print(f"prediction")
    print(f"p.shape: {p.shape}")
    print(f"p: {p}")

    e = rmse(p, y)
    print(f"rmse: {e}")

    ################ default prediction

    p2 = np.dot(x, coef)
    print(f"prediction2")
    print(f"p2.shape: {p2.shape}")
    print(f"p2: {p2}")

    e2 = mse(c, x, y)
    print(f"mse (2): {e2}")

    plt.figure()
    plt.plot((p2 - y)**2, label="prediction2")
    plt.plot(y, label="y")
    plt.legend()
    
    mpl.rcParams['legend.fontsize'] = 12
    plt.show()


main()