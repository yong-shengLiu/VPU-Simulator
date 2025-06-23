import math

def cordic(theta, iterations=7):
    """
    cordic: cos and sin
    """
    # 預先計算 arctangent 表格
    atan_table = [math.atan(2**-i) for i in range(iterations)]
    
    # 初始向量（CORDIC rotation mode 用的 "unit vector"）
    x, y = 1.0, 0.0
    z = theta

    # CORDIC gain（會縮小，最後需補償）
    K = 1.0
    for i in range(iterations):
        K *= 1 / math.sqrt(1 + 2**(-2*i))

    # 進行迭代旋轉
    for i in range(iterations):
        di = 1.0 if z >= 0 else -1.0
        x_new = x - di * y * (2**-i)
        y_new = y + di * x * (2**-i)
        z -= di * atan_table[i]
        x, y = x_new, y_new

    # 輸出前記得補償 gain
    return x * K, y * K



angle_rad = math.radians(30)  # 30 度 = π/6
cos_val, sin_val = cordic(angle_rad)

print(f"CORDIC cos(30) = {cos_val:.6f}, sin(30) = {sin_val:.6f}")
print(f"MATH   cos(30) = {math.cos(angle_rad):.6f}, sin(30) = {math.sin(angle_rad):.6f}")



print(math.atan(1)*180/math.pi, math.atan(0.5)*180/math.pi, math.atan(0.25)*180/math.pi, math.atan(0.125)*180/math.pi)