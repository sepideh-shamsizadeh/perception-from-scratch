import numpy as np

def d1_kalman_filter(z_curr, x_prev, p_prev,q, r, f):
    x_pred = f*x_prev
    p_pred = f*f*p_prev+q
    k_curr = p_pred/(p_pred+r)
    x_new = x_pred+ k_curr*(z_curr-x_pred)
    p_new = (1-k_curr)*p_pred
    return x_new, p_new, k_curr, x_pred, p_pred

def cv_matrices(dt, q):
    F = np.array([[1, dt], [0, 1]])
    G = np.array([[0.5*dt**2], [dt]])
    Q = q*G@G.T
    return F, G, Q

def vector_kamlman_filter(z_curr, x_prev, p_prev,q, r, dt,u, H):
    F, G, Q = cv_matrices(dt, q)
    R = np.atleast_2d(r)
    x_pred = F@x_prev + G@u
    p_pred = F@p_prev@F.T+Q
    k = p_pred@H.T @ (np.linalg.inv(H@p_pred@H.T+R))
    x_new = x_pred + k@(z_curr-H@x_pred)
    IKH = np.eye(p_pred.shape[0])-k@H
    p_new = IKH@p_pred@IKH.T + k@R@k.T
    return x_pred, p_pred, k, x_new, p_new


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

    r = [9]
    z = [2.0, 3.7, 6.6, 7.9, 10.2, 12.1, 13.8, 16.4]
    q = 0.5
    u = np.array([[0.5]])          # (1,1) — was np.array([0.5]), shape (1,)
    x_curr = np.array([[0.0], [0.0]])   # shape (2,1), not (2,)
    z_curr = np.array([[z]])            # shape (1,1)    
    p_curr = np.eye(2)*500
    for i, z_curr in enumerate(z):
        x_pred, p_pred, k_curr, x_new, p_new = vector_kamlman_filter(z_curr, x_curr, p_curr, q, r, dt=1.0,u=u, H=np.array([[1, 0]]))
        print(f"n:{i+1},x_pred:{x_pred}, p_pred:{p_pred}, k_curr:{k_curr}, x_new:{x_new}, p_new:{p_new}")
        x_curr, p_curr = x_new, p_new



if __name__=="__main__":
    main()
