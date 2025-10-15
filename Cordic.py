import math
import numpy as np
import matplotlib.pyplot as plt

def cordic(theta, iterations=7):
    """
    cordic: cos and sin
    """
    # Precompute the arctangent table for each iteration
    atan_table = [math.atan(2**-i) for i in range(iterations)]
    
    # 初始向量（CORDIC rotation mode 用的 "unit vector"）
    x, y = 1.0, 0.0
    z = theta

    # CORDIC gain（會縮小，最後需補償）
    K = 1.0
    for i in range(iterations):
        K *= 1 / math.sqrt(1 + 2**(-2*i))
        print(f"K: {K}")


    # 進行迭代旋轉
    for i in range(iterations):
        di = 1.0 if z >= 0 else -1.0
        x_new = x - di * y * (2**-i)
        y_new = y + di * x * (2**-i)
        z -= di * atan_table[i]
        print(f"z: {z}")
        x, y = x_new, y_new

    # 輸出前記得補償 gain
    return x * K, y * K



def sigmoid(x):
    ''' 
    It returns 1/(1+exp(-x)). where the values lies between zero and one 
    '''

    return 1/(1+np.exp(-x))


def tanh(x):
    ''' 
    It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.
    '''

    return np.tanh(x)


def RELU(x):
    ''' 
    It returns zero if the input is less than zero otherwise it returns the given input. 
    '''

    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)

    return x1


def softmax(x):
    ''' 
    Compute softmax values for each sets of scores in x. 
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def selu(x, alpha = 1.6732, lambda_ = 1.0507):
    return np.where(x > 0, lambda_ * x, lambda_ * alpha * (np.exp(x) - 1))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


angle_rad = math.radians(30)  # 30 度 = π/6
cos_val, sin_val = cordic(angle_rad)

print(f"CORDIC cos(30) = {cos_val:.6f}, sin(30) = {sin_val:.6f}")
print(f"MATH   cos(30) = {math.cos(angle_rad):.6f}, sin(30) = {math.sin(angle_rad):.6f}")



print(math.atan(1)*180/math.pi, math.atan(0.5)*180/math.pi, math.atan(0.25)*180/math.pi, math.atan(0.125)*180/math.pi)



x = np.linspace(-10, 10)
plt.plot(x, gelu(x))
plt.axis('tight')
plt.title('Activation Function :GELU')
plt.show()

x = np.linspace(-10, 10)
plt.plot(x, sigmoid(x))
plt.axis('tight')
plt.title('Activation Function :Sigmoid')
plt.show()


x = np.linspace(-10, 10)
plt.plot(x, tanh(x))
plt.axis('tight')
plt.title('Activation Function :Tanh')
plt.show()


x = np.linspace(-10, 10)
plt.plot(x, RELU(x))
plt.axis('tight')
plt.title('Activation Function :RELU')
plt.show()


x = np.linspace(-10, 10)
plt.plot(x, softmax(x))
plt.axis('tight')
plt.title('Activation Function :Softmax')
plt.show()

x = np.linspace(-10, 10)
plt.plot(x, selu(x))
plt.axis('tight')
plt.title('Activation Function :SELU')
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Define CORDIC parameters
# ITERATIONS = 20

# # Hyperbolic arctangent lookup table (skip i=4 and i=13 for convergence)
# atanh_table = [np.arctanh(2**-i) for i in range(ITERATIONS) if i != 4 and i != 13]

# # Hyperbolic CORDIC gain factor (product of scale factors)
# KH = np.prod([1 / np.sqrt(1 - 2**(-2 * i)) for i in range(ITERATIONS) if i != 4 and i != 13])

# def cordic_hyperbolic_rotation(z, iterations=ITERATIONS):
#     """
#     Hyperbolic CORDIC rotation to compute sinh(z) and cosh(z).
#     Returns (sinh(z), cosh(z)).
#     """
#     x = KH  # scaled cosh(z)
#     y = 0.0
#     zi = z
#     idx = 0

#     for i in range(iterations):
#         if i == 4 or i == 13:
#             continue  # skip these iterations for hyperbolic mode

#         di = 1.0 if zi >= 0 else -1.0
#         x_new = x + di * y * (2 ** -i)
#         y_new = y + di * x * (2 ** -i)
#         zi -= di * atanh_table[idx]

#         x, y = x_new, y_new
#         idx += 1

#     return y, x  # sinh(z), cosh(z)

# def cordic_tanh(x):
#     """
#     Computes tanh(x) using CORDIC hyperbolic rotation.
#     """
#     sinh_x, cosh_x = cordic_hyperbolic_rotation(x)
#     return sinh_x / cosh_x

# def cordic_sigmoid(x):
#     """
#     Computes sigmoid(x) using the identity:
#     sigmoid(x) = (tanh(x/2) + 1) / 2
#     """
#     return 0.5 * (cordic_tanh(x / 2) + 1)

# # Values to plot
# x_vals = np.linspace(-5, 5, 300)
# cordic_tanh_vals = np.array([cordic_tanh(x) for x in x_vals])
# cordic_sigmoid_vals = np.array([cordic_sigmoid(x) for x in x_vals])
# true_tanh_vals = np.tanh(x_vals)
# true_sigmoid_vals = 1 / (1 + np.exp(-x_vals))

# # Plot
# plt.figure(figsize=(10, 5))

# # Plot tanh
# plt.subplot(1, 2, 1)
# plt.plot(x_vals, true_tanh_vals, 'k--', label='True tanh')
# plt.plot(x_vals, cordic_tanh_vals, 'b-', label='CORDIC tanh')
# plt.title("CORDIC tanh vs True tanh")
# plt.xlabel("x")
# plt.ylabel("tanh(x)")
# plt.grid(True)
# plt.legend()

# # Plot sigmoid
# plt.subplot(1, 2, 2)
# plt.plot(x_vals, true_sigmoid_vals, 'k--', label='True sigmoid')
# plt.plot(x_vals, cordic_sigmoid_vals, 'g-', label='CORDIC sigmoid')
# plt.title("CORDIC sigmoid vs True sigmoid")
# plt.xlabel("x")
# plt.ylabel("sigmoid(x)")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()