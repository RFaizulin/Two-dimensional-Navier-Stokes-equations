import numpy as np


class RK4Solver:
    calculate_a = lambda self, T: np.sqrt(self.gamma * self.R * T)
    calculate_mu = lambda self, T: self.mu0 * ((T/self.T0)**(3/2)) * (self.T0 + 110) / (T + 110)
    calculate_rho = lambda self, p, T: p / (self.R * T)
    calculate_k = lambda self, T: self.calculate_mu(T) * self.cp / self.Pr

    def __init__(self, grid_sizes=(70, 70), M_inf=4, T_inf=288.16, p_inf=101325, cfl=0.6, case='constant'):
        self.x_grid_size, self.y_grid_size = grid_sizes
        self.nx, self.ny = self.x_grid_size + 1, self.y_grid_size + 1

        self.gamma = 1.4
        self.Pr = 0.71
        self.R = 287
        self.mu0 = 1.7894*1e-5
        self.T0 = 288.16
        self.cfl = cfl

        self.M_inf = M_inf
        self.T_inf = T_inf
        self.p_inf = p_inf
        self.a_inf = self.calculate_a(T_inf)
        self.u_inf = M_inf * self.a_inf
        self.v_inf = 0
        self.LHORI = 1e-5

        self.T_w = np.copy(T_inf)
        self.mu_inf = self.calculate_mu(T_inf)

        self.cv = self.R / (self.gamma-1)
        self.cp = self.gamma * self.cv
        self.rho_inf = self.calculate_rho(p_inf, T_inf)
        self.Re_l = (self.rho_inf * self.u_inf * self.LHORI) / self.calculate_mu(T_inf)
        self.e_inf = self.cv * T_inf
        self.k_inf = self.calculate_k(T_inf)

        self.delta = (5 * self.LHORI) / np.sqrt(self.Re_l)
        self.LVERT = 5 * self.delta
        self.dx = self.LHORI / self.x_grid_size
        self.dy = self.LVERT / self.y_grid_size
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dy
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')

        self.n_vars = 5
        self.n_eq = 4
        self.case = case
        self.initial_data = np.array([
            self.rho_inf,
            self.u_inf,
            self.v_inf,
            p_inf,
            T_inf
        ])

    def get_xx(self):
        return self.xx

    def get_yy(self):
        return self.yy

    def initial_condition(self):
        V = np.zeros((self.n_vars, self.nx, self.ny))
        for i in range(self.n_vars):
            V[i] = self.initial_data[i]
        return V

    def boundary_condition(self, V):
        # case 1 (leading edge)
        # i = 0, j = 0
        s = np.index_exp[0, 0]
        V[1][s] = 0
        V[2][s] = 0
        V[3][s] = self.p_inf
        V[4][s] = self.T_inf
        V[0][s] = self.rho_inf

        # case 2 (inflow/upper boundary - not leading edge)
        # i = 0, j = 1,ny
        s = np.index_exp[0, 1:]
        for i in range(self.n_vars):
            V[i][s] = self.initial_data[i]
        # i = 0,nx, j = ny
        s = np.index_exp[:, -1]
        for i in range(self.n_vars):
            V[i][s] = self.initial_data[i]

        # case 3 (surface - not leading edge)
        # i = 1,nx, j = 0
        s = np.index_exp[1:, 0]
        V[1][s] = 0
        V[2][s] = 0
        V[3][s] = 2*V[3][1:, 1] - V[3][1:, 2]
        if self.case == 'constant':
            # T(i, 0) = T_wall
            V[4][s] = self.T_w
        elif self.case == 'adiabatic':
            # dT/dy(i, 0) = (T(i, 1) - T(i, 0)) / dy = 0
            # T(i, 0) = T(i, 1)
            V[4][1:, 0] = V[4][1:, 1]
        V[0][s] = self.calculate_rho(V[3][s], V[4][s])

        # case 4 (outflow - not surface)
        # i = nx, j = 1,ny-1
        s = np.index_exp[-1, 1:-1]
        for i in range(1, self.n_vars):
            # u(nx, j) = 2*u(nx-1, j) - u(nx-2, j)
            V[i][s] = 2*V[i][-2, 1:-1] - V[i][-3, 1:-1]
        V[0][s] = self.calculate_rho(V[3][s], V[4][s])

        return V

    def transform_V_to_U(self, V):
        U = np.zeros((self.n_eq, self.nx, self.ny))
        U[0] = V[0]
        U[1] = V[0] * V[1]
        U[2] = V[0] * V[2]
        velocity2 = np.square(V[1]) + np.square(V[2])
        U[3] = V[0] * (self.cv * V[4] + velocity2 / 2)
        return U

    def transform_U_to_V(self, U):
        V = np.zeros((self.n_vars, self.nx, self.ny))
        V[0] = U[0]
        V[1] = U[1] / U[0]
        V[2] = U[2] / U[0]
        velocity2 = np.square(V[1]) + np.square(V[2])
        V[4] = 1/self.cv * (U[3] / U[0] - velocity2 / 2)
        V[3] = V[0] * self.R * V[4]
        return V

    def timestep(self, V):
        rho, u, v, T = V[0], V[1], V[2], V[4]
        a = self.calculate_a(T)
        mu = self.calculate_mu(T)
        k1 = (1/np.square(self.dx) + 1/np.square(self.dy))
        k2 = np.sqrt(k1)
        v_pr = np.max(4/3*mu * (self.gamma*mu/self.Pr) / rho)

        dt_cfl = 1 / (np.abs(u)/self.dx + np.abs(v)/self.dy + a*k2 + 2*v_pr*k1)
        dt = np.min(self.cfl * dt_cfl)
        return dt

    def calculate_D(self, u, x_derivative=True):
        D = np.zeros((self.nx, self.ny))
        if x_derivative:
            D[0, :] = (u[1, :] - u[0, :]) / self.dx
            D[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*self.dx)
            D[-1, :] = (u[-1, :] - u[-2, :]) / self.dx
        else:
            D[:, 0] = (u[:, 1] - u[:, 0]) / self.dy
            D[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*self.dy)
            D[:, -1] = (u[:, -1] - u[:, -2]) / self.dy
        return D

    def calculate_flow(self, xi, u, x_derivative=True):
        D = np.zeros((self.nx, self.ny))
        if x_derivative:
            ur = 0.5*(u[2:, :] + u[1:-1, :])
            ul = 0.5*(u[1:-1, :] + u[:-2, :])
            xi_r_ur = 0.5*xi[1:-1, :]*(ur+np.abs(ur)) + 0.5*xi[2:, :]*(ur-np.abs(ur))
            xi_l_ul = 0.5*xi[:-2, :]*(ul+np.abs(ul)) + 0.5*xi[1:-1, :]*(ul-np.abs(ul))
            D[1:-1, :] = (xi_r_ur - xi_l_ul) / self.dx
        else:
            ur = 0.5*(u[:, 2:] + u[:, 1:-1])
            ul = 0.5*(u[:, 1:-1] + u[:, :-2])
            xi_r_ur = 0.5*xi[:, 1:-1]*(ur+np.abs(ur)) + 0.5*xi[:, 2:]*(ur-np.abs(ur))
            xi_l_ul = 0.5*xi[:, :-2]*(ul+np.abs(ul)) + 0.5*xi[:, 1:-1]*(ul-np.abs(ul))
            D[:, 1:-1] = (xi_r_ur - xi_l_ul) / self.dy
        return D

    def calculate_R(self, U, V):
        U1, U2, U3, U5 = U
        u, v, p, T = V[1:]

        mu = self.calculate_mu(T)
        _lambda = (-2/3) * mu
        k = self.calculate_k(T)
        Dy_u = self.calculate_D(u, x_derivative=False)
        Dx_v = self.calculate_D(v)
        Dx_u = self.calculate_D(u)
        Dy_v = self.calculate_D(v, x_derivative=False)
        tau_xy = mu * (Dy_u + Dx_v)
        div = Dx_u + Dy_v
        tau_xx = _lambda * div + 2*mu*Dx_u
        tau_yy = _lambda * div + 2*mu*Dy_v
        Dx_T = self.calculate_D(T)
        Dy_T = self.calculate_D(T, x_derivative=False)
        qx = -k*Dx_T
        qy = -k*Dy_T
        Dx_p = self.calculate_D(p)
        Dy_p = self.calculate_D(p, x_derivative=False)

        # continuity
        Dx_rho_u = self.calculate_flow(U1, u)

        Dy_rho_v = self.calculate_flow(U1, v, x_derivative=False)

        Dt_rho = -(Dx_rho_u + Dy_rho_v)

        # x momentum
        Dx_rho_uu = self.calculate_flow(U2, u)
        Dx_tau_xx = self.calculate_D(tau_xx)

        Dy_rho_uv = self.calculate_flow(U2, v, x_derivative=False)
        Dy_tau_yx = self.calculate_D(tau_xy, x_derivative=False)

        Dt_rho_u = -(Dx_rho_uu + Dx_p - Dx_tau_xx) -            (Dy_rho_uv - Dy_tau_yx)

        # y momentum
        Dx_rho_uv = self.calculate_flow(U3, u)
        Dx_tau_xy = self.calculate_D(tau_xy)

        Dy_rho_vv = self.calculate_flow(U3, v, x_derivative=False)
        Dy_tau_yy = self.calculate_D(tau_yy, x_derivative=False)

        Dt_rho_v = -(Dx_rho_uv - Dx_tau_xy) -            (Dy_rho_vv + Dy_p - Dy_tau_yy)

        # energy
        Dx_Et_u = self.calculate_flow(U5, u)
        Dx_pu = self.calculate_flow(p, u)
        Dx_qx = self.calculate_D(qx)
        Dx_u_tau_xx = self.calculate_D(u*tau_xx)
        Dx_v_tau_xy = self.calculate_D(v*tau_xy)

        Dy_Et_v = self.calculate_flow(U5, v, x_derivative=False)
        Dy_pv = self.calculate_flow(p, v, x_derivative=False)
        Dy_qy = self.calculate_D(qy, x_derivative=False)
        Dy_u_tau_yx = self.calculate_D(u*tau_xy, x_derivative=False)
        Dy_v_tau_yy = self.calculate_D(v*tau_yy, x_derivative=False)

        Dt_Et = -(Dx_Et_u + Dx_pu + Dx_qx - Dx_u_tau_xx - Dx_v_tau_xy) -            (Dy_Et_v + Dy_pv + Dy_qy - Dy_u_tau_yx - Dy_v_tau_yy)

        R = np.array([
            Dt_rho,
            Dt_rho_u,
            Dt_rho_v,
            Dt_Et
        ])
        return R

    def rk4_method(self, U, V, dt):
        Rn = self.calculate_R(U, V)
        # step 1
        U1 = U + 0.5*dt*Rn
        R1 = self.calculate_R(U1, V)
        # step 2
        U2 = U + 0.5*dt*R1
        R2 = self.calculate_R(U2, V)
        # step 3
        U3 = U + dt*R2
        R3 = self.calculate_R(U3, V)
        # step 4
        U_new = U + (Rn + 2*R1 + 2*R2 + R3)*dt/6
        return U_new

    def solve(self, tolerance=1e-8, Vc=dict()):
        n = 0
        if len(Vc) == 0:
            V_old = self.initial_condition()
            t = 0
        else:
            V_old = Vc['V_old']
            t = Vc['t']
        converge = True

        while converge:
            dt = self.timestep(V_old)
            t += dt
            n += 1
            rho_old = V_old[0]

            U_old = self.transform_V_to_U(V_old)
            U_new = self.rk4_method(U_old, V_old, dt)
            V_new = self.transform_U_to_V(U_new)
            V_new = self.boundary_condition(V_new)
            V_old = np.copy(V_new)

            rho_new = V_new[0]
            change = np.max(np.abs(rho_new - rho_old))
            if n == 1:
                converge = True
            else:
                converge = False if change <= tolerance else True

            if n % 1000 == 0:
                print(change)
                np.savez(f'V_{change}', V_old=V_old, t=t)
        return V_new
    
    
# constant case
solver = RK4Solver(grid_sizes=(800, 800), cfl=0.1)
V = solver.solve()
np.save('V_constant', V)
np.save('grid_constant', np.array([solver.xx, solver.yy]))
y, x = solver.y, solver.x
Re_l = solver.Re_l
y_norm = y / x[-1] * np.sqrt(Re_l)
u_te = V[1][-1, :] / solver.u_inf
p_te = V[3][-1, :] / solver.p_inf
t_te = V[4][-1, :] / solver.T_inf
np.save('yupT_constant', np.array([y_norm, u_te, p_te, t_te]))


# adiabatic case
solver = RK4Solver(grid_sizes=(800, 800), cfl=0.1, case='adiabatic')
V = solver.solve()
np.save('V_adiabatic', V)
np.save('grid_adiabatic', np.array([solver.xx, solver.yy]))
y, x = solver.y, solver.x
Re_l = solver.Re_l
y_norm = y / x[-1] * np.sqrt(Re_l)
u_te = V[1][-1, :] / solver.u_inf
p_te = V[3][-1, :] / solver.p_inf
t_te = V[4][-1, :] / solver.T_inf
np.save('yupT_adiabatic', np.array([y_norm, u_te, p_te, t_te]))