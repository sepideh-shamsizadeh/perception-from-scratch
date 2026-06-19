import numpy as np

def d1_kalman_filter(z_curr, x_prev, p_prev,q, r, f):
    x_pred = f*x_prev
    p_pred = f*f*p_prev+q
    k_curr = p_pred/(p_pred+r)
    x_new = x_pred+ k_curr*(z_curr-x_pred)
    p_new = (1-k_curr)*p_pred
    return x_new, p_new, k_curr, x_pred, p_pred


def main():
    x_curr = 60
    p_curr = 225
    r= 25
    q = 4
    f = 1
    z = [48.5, 47.2, 55 , 49.8, 50.6]
    for i, z_curr in enumerate(z):
        x_new, p_new, k_curr, x_pred, p_pred = d1_kalman_filter(z_curr, x_curr, p_curr, q, r, f)
        print(f"n:{i+1},x_pred:{x_pred:.4f}, p_pred:{p_pred:.4f}, k_curr:{k_curr:.4f}, x_new:{x_new:.4f}, p_new:{p_new:.2f}")
        x_curr, p_curr = x_new, p_new


if __name__=="__main__":
    main()
