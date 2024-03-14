import numpy as np

class Collision():
    @staticmethod
    def lenard_bernstein(sys, s, i, j, k):
        N = sys.species[s].num_modes - 1
        coll_i = i*(i-1)*(i-2)/(N[0]*(N[0]-1)*(N[0]-2)) if N[0]>2 else 0
        coll_j = j*(j-1)*(j-2)/(N[1]*(N[1]-1)*(N[1]-2)) if N[1]>2 else 0
        coll_k = k*(k-1)*(k-2)/(N[2]*(N[2]-1)*(N[2]-2)) if N[2]>2 else 0
        return - sys.species[s].collision_rate * sys.C(s, i, j, k) * (coll_i + coll_j + coll_k)

    @staticmethod
    def BGK(sys):
        raise NotImplementedError("BGK operator not yet implemented")


class Advection:
    @staticmethod
    def cartesian(sys, s, i, j, k):
        hz = (sys.domain['R'] - sys.domain['L']) / sys.domain['N']
        u = sys.shift(s)
        du = sys.shift_grad(s)
        a = sys.scale(s)
        da = sys.scale_grad(s)
        terms = 0
        # Terms for spatially constant Hermite parameters
        terms += sys.C_grad(s,i,j,k+1) * np.sqrt((k+1)/2)*a[2]
        terms += sys.C_grad(s,i,j,k) * u[2]
        terms += sys.C_grad(s,i,j,k-1) * np.sqrt(k/2)*a[2]
        # Additional terms for spatially varying Hermite parameters (u, a varying)
        terms += sys.C(s,i,j,k+1) * (k+2)*np.sqrt((k+1)/2)*da[2]
        terms += sys.C(s,i,j,k) * (k+1)*( du[2] + u[2]*da[2]/a[2] )
        terms += sys.C(s,i,j,k-1) * ( k*np.sqrt(k/2)*da[2] + np.sqrt(2*k)*u[2]*du[2]/a[2] + (k+1)*np.sqrt(k/2)*da[2] ) 
        terms += sys.C(s,i,j,k-2) * np.sqrt(k*(k-1))*( u[2]*da[2]/a[2] + du[2] )
        terms += sys.C(s,i,j,k-3) * np.sqrt(k*(k-1)*(k-2)/2)*da[2]

        return terms

    @staticmethod
    def cylindrical(sys, s, i, j, k):
        raise NotImplementedError("Cylindrical advection operator not implemented!")

class Acceleration():
    @staticmethod
    def electric(sys, s, i, j, k, E): # Only one component as model is 1D and therefore Ey=0, Ez=0
        terms = - E * sys.species[s].q/sys.species[s].m * np.sqrt(2*k) / sys.species[s].scale[2] * sys.C(s, i, j, k-1)
        return terms

    @staticmethod
    def magnetic(sys, s, i, j, k, B):
        u = sys.shift(s)
        du = sys.shift_grad(s)
        a = sys.scale(s)
        da = sys.scale_grad(s)
        terms = 0
        # Component 1
        terms += - sys.species[s].q/sys.species[s].m * B[2] * np.sqrt(2*i)/a[0] * (a[1]*np.sqrt((j+1)/2)*sys.C(s,i-1,j+1,k) + u[1]*sys.C(s,i-1,j,k) + a[1]*np.sqrt(j/2)*sys.C(s,i-1,j-1,k))
        terms +=   sys.species[s].q/sys.species[s].m * B[0] * np.sqrt(2*i)/a[0] * (a[2]*np.sqrt((k+1)/2)*sys.C(s,i-1,j,k+1) + u[2]*sys.C(s,i-1,j,k) + a[2]*np.sqrt(k/2)*sys.C(s,i-1,j,k-1))
        # Component 2
        terms += - sys.species[s].q/sys.species[s].m * B[0] * np.sqrt(2*j)/a[1] * (a[2]*np.sqrt((k+1)/2)*sys.C(s,i,j-1,k+1) + u[2]*sys.C(s,i,j-1,k) + a[2]*np.sqrt(k/2)*sys.C(s,i,j-1,k-1))
        terms +=   sys.species[s].q/sys.species[s].m * B[2] * np.sqrt(2*j)/a[1] * (a[0]*np.sqrt((i+1)/2)*sys.C(s,i+1,j-1,k) + u[0]*sys.C(s,i,j-1,k) + a[0]*np.sqrt(i/2)*sys.C(s,i-1,j-1,k))
        # Component 3
        terms += - sys.species[s].q/sys.species[s].m * B[1] * np.sqrt(2*k)/a[2] * (a[0]*np.sqrt((i+1)/2)*sys.C(s,i+1,j,k-1) + u[0]*sys.C(s,i,j,k-1) + a[0]*np.sqrt(i/2)*sys.C(s,i-1,j,k-1))
        terms +=   sys.species[s].q/sys.species[s].m * B[0] * np.sqrt(2*k)/a[2] * (a[1]*np.sqrt((j+1)/2)*sys.C(s,i,j+1,k-1) + u[1]*sys.C(s,i,j,k-1) + a[1]*np.sqrt(j/2)*sys.C(s,i,j-1,k-1))
        
        return terms        

    @staticmethod
    def centripetal(sys, s, i, j, k):
        u = sys.shift(s)
        du = sys.shift_grad(s)
        a = sys.scale(s)
        da = sys.scale_grad(s)
        terms = 0
        terms += sys.C(s,i-1,j+2,k) * a[1]**2/2 * np.sqrt((j+1)*(j+2))
        terms += sys.C(s,i-1,j-1,k) * u[1]*a[1] * ( np.sqrt((j+1)/2) + np.sqrt(2*j))
        terms += sys.C(s,i-1,j,k) * ( u[1]**2 + a[1]**2*(j+1/2) )
        terms += sys.C(s,i-1,j+1,k) * u[1]*a[1] * np.sqrt((j+1)/2)
        terms += sys.C(s,i-1,j-2,k) * a[1]**2/2 * np.sqrt(j*(j-1))
        terms *= - 1/r*np.sqrt(2*i)/a[0]
        return terms

    @staticmethod
    def coriolis(sys, s, i, j, k):
        u = sys.shift(s)
        du = sys.shift_grad(s)
        a = sys.scale(s)
        da = sys.scale_grad(s)
        terms = 0
        terms += sys.C(s,i+1,j,k) * (j+1) * np.sqrt((i+1)/2) * a[0]
        terms += sys.C(s,i,j,k) * (j+1) * u[0]
        terms += sys.C(s,i-1,j,k) * (j+1) * np.sqrt(i/2) * a[0]
        terms += sys.C(s,i+1,j-1,k) * np.sqrt(j*(j+1)) * a[0]/a[1]*u[1]
        terms += sys.C(s,i-1,j,k) * np.sqrt(2*j) * u[1]/a[1]*u[0]
        terms += sys.C(s,i-1,j-1,k) * np.sqrt(i*j) * a[0]/a[1]
        terms += sys.C(s,i+1,j-2,k) * np.sqrt(j*(j-1)*(i+1)/2) * a[0]
        terms += sys.C(s,i,j-2,k) * np.sqrt(j*(j-1)) * u[0]
        terms += sys.C(s,i-1,j-2,k) * np.sqrt(i*j*(j-1)/2) * a[0]
        terms *= 2/r
        return terms

class KineticEquation(): # Implementation of kinetic equation in form dC/dt = terms
    @staticmethod
    def cartesian(sys, s, i, j, k, E=0, B=0):
        terms = 0
        terms -= Advection.cartesian(sys, s, i, j, k)
        terms -= Acceleration.electric(sys, s, i, j, k, E)
        terms -= Acceleration.magnetic(sys, s, i, j, k, B)
        terms += Collision.lenard_bernstein(sys, s, i, j, k)
        return terms

    @staticmethod
    def cylindrical(sys, s, i, j, k, E=0, B=0):
        terms = 0
        terms -= Advection.cylindrical(sys, s, i, j, k)
        terms -= Acceleration.electric(sys, s, i, j, k, E)
        terms -= Acceleration.magnetic(sys, s, i, j, k, B)
        terms -= Acceleration.centripetal(sys, s, i, j, k)
        terms -= Acceleration.coriolis(sys, s, i, j, k)
        terms += Collision.lenard_bernstein(sys, s, i, j, k)
        return terms

class Time(): # Implicit time discretization operators in format f=0
    @staticmethod
    def implicit_euler(sys_curr, sys_next, K, dt, s, i, j, k, E, B):
        # Implicit Euler: dC/dt = K(C) -> C(t+dt) = C(t) + K(C(t+dt))*dt with K being a nonlinear operator representing the kinetic equation
        # Therefore, the equation to solve is C(t+dt) - C(t) - K(C(t+dt))*dt = 0
        C_curr = sys_curr.C(s, i, j, k)
        C_next = sys_next.C(s, i, j, k)
        K_next = K(sys_next, s, i, j, k, E, B)

        return C_next - C_curr - K_next * dt

    @staticmethod
    def crank_nicolson(sys_curr, sys_next, K, dt, s, i, j, k, E, B):
        # Crank-Nicolson: dC/dt = K(C) -> C(t+dt) = C(t) + 0.5(K(C(t) + K(C(t+dt)))*dt with K being a nonlinear operator representing the kinetic equation
        # Therefore, the equation to solve is C(t+dt) - C(t) - 0.5(K(C(t) + K(C(t+dt)))*dt = 0
        C_curr = sys_curr.C(s, i, j, k)
        C_next = sys_next.C(s, i, j, k)
        K_curr = K(sys_curr, s, i, j, k, E, B)
        K_next = K(sys_next, s, i, j, k, E, B)
        return C_next - C_curr - 0.5 * (K_curr + K_next) * dt