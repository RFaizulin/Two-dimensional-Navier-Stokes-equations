import numpy as np


class MacSolver:
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
        self.x = np.arange(self.nx) * self.dx + 1e-10
        self.y = np.arange(self.ny) * self.dy 
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        
        self.n_vars = 5
        self.n_eq = 4
        self.tf = 0
        self.n_iters = 0
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
        
    def calculate_DxE_DyF(self, E, F, right_derivative=True):
        Dx_E = np.zeros((self.n_eq, self.nx, self.ny))
        Dy_F = np.zeros((self.n_eq, self.nx, self.ny))
        if right_derivative:
            # dE/dx(i, j, t) = (E(i+1, j, t) - E(i, j, t)) / dx
            # i = 0,nx-1, j = 0,ny
            Dx_E[:, :-1, :] = (E[:, 1:, :] - E[:, :-1, :]) / self.dx
            # dF/dy(i, j, t) = (F(i, j+1, t) - F(i, j, t)) / dy
            # i = 0,nx, j = 0,ny-1
            Dy_F[:, :, :-1] = (F[:, :, 1:] - F[:, :, :-1]) / self.dy
        else:
            # dE/dx(i, j, t) = (E(i, j, t) - E(i-1, j, t)) / dx
            # i = 1,nx, j = 0,ny
            Dx_E[:, 1:, :] = (E[:, 1:, :] - E[:, :-1, :]) / self.dx
            # dF/dy(i, j, t) = (F(i, j, t) - F(i, j-1, t)) / dy
            # i = 0,nx, j = 1,ny
            Dy_F[:, :, 1:] = (F[:, :, 1:] - F[:, :, :-1]) / self.dy
        return Dx_E, Dy_F
    
    def calculate_D_left(self, u, x_derivative=True):
        D = np.zeros((self.nx, self.ny))
        if x_derivative:
            D[0, :] = (u[1, :] - u[0, :]) / self.dx
            # du/dx(i, j) = (u(i, j) - u(i-1, j)) / dx
            # i = 1,nx, j = 0,ny
            D[1:, :] = (u[1:, :] - u[:-1, :]) / self.dx
        else:
            D[:, 0] = (u[:, 1] - u[:, 0]) / self.dy
            # du/dy(i, j) = (u(i, j) - u(i, j-1)) / dy
            # i = 0,nx, j = 1,ny
            D[:, 1:] = (u[:, 1:] - u[:, :-1]) / self.dy
        return D
    
    def calculate_D_right(self, u, x_derivative=True):
        D = np.zeros((self.nx, self.ny))
        if x_derivative:
            # du/dx(i, j) = (u(i+1, j) - u(i, j)) / dx
            # i = 0,nx-1, j = 0,ny
            D[:-1, :] = (u[1:, :] - u[:-1, :]) / self.dx
            D[-1, :] = (u[-1, :] - u[-2, :]) / self.dx
        else:
            # du/dy(i, j) = (u(i, j+1) - u(i, j)) / dy
            # i = 0,nx, j = 0,ny-1
            D[:, :-1] = (u[:, 1:] - u[:, :-1]) / self.dy
            D[:, -1] = (u[:, -1] - u[:, -2]) / self.dy
        return D
    
    def calculate_D_central(self, u, x_derivative=True):
        D = np.zeros((self.nx, self.ny))
        if x_derivative:
            D[0, :] = (u[1, :] - u[0, :]) / self.dx
            # du/dx(i, j) = (u(i+1, j) - u(i-1, j)) / 2dx
            # i = 1,nx-1, j = 0,ny
            D[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2*self.dx)
            D[-1, :] = (u[-1, :] - u[-2, :]) / self.dx
        else:
            D[:, 0] = (u[:, 1] - u[:, 0]) / self.dy
            # du/dy(i, j) = (u(i, j+1) - u(i, j-1)) / 2dy
            # i = 0,nx, j = 1,ny-1
            D[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*self.dy)
            D[:, -1] = (u[:, -1] - u[:, -2]) / self.dy
        return D
    
    def mac_cormack_method(self, U, V, dt):
        U1, U2, U3, U5 = U
        u, v, p, T = V[1:]
        
        # predictor step
        mu = self.calculate_mu(T)
        _lambda = (-2/3) * mu
        k = self.calculate_k(T)
        
        # dE/dx (right)
        # Dx
        Dx_v = self.calculate_D_left(v)
        Dx_u = self.calculate_D_left(u)
        Dx_T = self.calculate_D_left(T)
        # Dy
        Dy_u = self.calculate_D_central(u, x_derivative=False)
        Dy_v = self.calculate_D_central(v, x_derivative=False)
        
        tau_xy = mu * (Dy_u + Dx_v)
        tau_xx = _lambda * (Dx_u + Dy_v) + 2*mu*Dx_u
        qx = -k*Dx_T
        E = np.array([U1*u,
                     U2*u + p - tau_xx,
                     U2*v - tau_xy,
                     (U5+p)*u - u*tau_xx - v*tau_xy + qx])
        # dF/dy (right)
        # Dy
        Dy_u = self.calculate_D_left(u, x_derivative=False)
        Dy_v = self.calculate_D_left(v, x_derivative=False)
        Dy_T = self.calculate_D_left(T, x_derivative=False)
        # Dx
        Dx_v = self.calculate_D_central(v)
        Dx_u = self.calculate_D_central(u)
        
        tau_xy = mu * (Dy_u + Dx_v)
        tau_yy = _lambda * (Dx_u + Dy_v) + 2*mu*Dy_v
        qy = -k*Dy_T
        F = np.array([U1*v,
                     U3*u - tau_xy,
                     U3*v + p - tau_yy,
                     (U5+p)*v - u*tau_xy - v*tau_yy + qy])
        
        Dx_E, Dy_F = self.calculate_DxE_DyF(E, F)
        U_star = U - dt*Dx_E - dt*Dy_F
        
        # corrector step
        V_star = self.transform_U_to_V(U_star)
        V_star = self.boundary_condition(V_star)
        rho, u, v, p, T = V_star
        e = self.cv * T
        velocity2 = np.square(u) + np.square(v)
        Et = rho * (e + velocity2/2)
        
        mu = self.calculate_mu(T)
        _lambda = (-2/3) * mu
        k = self.calculate_k(T)
        
        # dE/dx (left)
        # Dx
        Dx_v = self.calculate_D_right(v)
        Dx_u = self.calculate_D_right(u)
        Dx_T = self.calculate_D_right(T)
        # Dy
        Dy_u = self.calculate_D_central(u, x_derivative=False)
        Dy_v = self.calculate_D_central(v, x_derivative=False)
        
        tau_xy = mu * (Dy_u + Dx_v)
        tau_xx = _lambda * (Dx_u + Dy_v) + 2*mu*Dx_u
        qx = -k*Dx_T
        E = np.array([rho*u,
                     rho*np.square(u) + p - tau_xx,
                     rho*u*v - tau_xy,
                     (Et+p)*u - u*tau_xx - v*tau_xy + qx])
        # dF/dy (left)
        # Dy
        Dy_u = self.calculate_D_right(u, x_derivative=False)
        Dy_v = self.calculate_D_right(v, x_derivative=False)
        Dy_T = self.calculate_D_right(T, x_derivative=False)
        # Dx
        Dx_v = self.calculate_D_central(v)
        Dx_u = self.calculate_D_central(u)
        
        tau_xy = mu * (Dy_u + Dx_v)
        tau_yy = _lambda * (Dx_u + Dy_v) + 2*mu*Dy_v
        qy = -k*Dy_T
        F = np.array([rho*v,
                     rho*u*v - tau_xy,
                     rho*np.square(v) + p - tau_yy,
                     (Et+p)*v - u*tau_xy - v*tau_yy + qy])
        
        Dx_E, Dy_F = self.calculate_DxE_DyF(E, F, right_derivative=False)
        U_new = 0.5*(U + U_star - dt*Dx_E - dt*Dy_F)   
        return U_new
    
    def solve(self, tolerance=1e-8):
        V_old = self.initial_condition()
        converge = True
        
        while converge:
            dt = self.timestep(V_old)
            self.tf += dt
            self.n_iters += 1
            rho_old = V_old[0]
            
            U_old = self.transform_V_to_U(V_old)
            U_new = self.mac_cormack_method(U_old, V_old, dt)
            V_new = self.transform_U_to_V(U_new)
            V_new = self.boundary_condition(V_new)
            V_old = np.copy(V_new)
            
            rho_new = V_new[0]
            change = np.max(np.abs(rho_new - rho_old))
            converge = False if change <= tolerance else True
            if self.n_iters % 500 == 0:
                print(change)
        return V_new
    
    
# constant case
solver = MacSolver()
V = solver.solve()
np.save('V_constant_mac', V)
np.save('grid_constant_mac', np.array([solver.xx, solver.yy]))
y, x = solver.y, solver.x
Re_l = solver.Re_l
y_norm = y / x[-1] * np.sqrt(Re_l)
u_te = V[1][-1, :] / solver.u_inf
p_te = V[3][-1, :] / solver.p_inf
t_te = V[4][-1, :] / solver.T_inf
np.save('yupT_constant_mac', np.array([y_norm, u_te, p_te, t_te]))


# adiabatic case
solver = MacSolver(cfl=0.5, case='adiabatic')
V = solver.solve()
np.save('V_adiabatic_mac', V)
np.save('grid_adiabatic_mac', np.array([solver.xx, solver.yy]))
y, x = solver.y, solver.x
Re_l = solver.Re_l
y_norm = y / x[-1] * np.sqrt(Re_l)
u_te = V[1][-1, :] / solver.u_inf
p_te = V[3][-1, :] / solver.p_inf
t_te = V[4][-1, :] / solver.T_inf
np.save('yupT_adiabatic_mac', np.array([y_norm, u_te, p_te, t_te]))
