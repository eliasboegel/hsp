import numpy as np

class Collision():
    @staticmethod
    def lenard_bernstein(sys, s, i, j, k):
        N = sys.species[s].num_modes - 1
        coll_i = i*(i-1)*(i-2)/(N[0]*(N[0]-1)*(N[0]-2)) if N[0]>2 else 0
        coll_j = j*(j-1)*(j-2)/(N[1]*(N[1]-1)*(N[1]-2)) if N[1]>2 else 0
        coll_k = k*(k-1)*(k-2)/(N[2]*(N[2]-1)*(N[2]-2)) if N[2]>2 else 0
        return - sys.species[s].collision_rate * sys.C(s, i, j, k) * (coll_i + coll_j + coll_k)

class Advection:
    @staticmethod
    def cartesian(sys, s, i, j, k):
        terms = 0
        terms += sys.dC(s, i, j, k)   * sys.species[s].shift[2]
        terms += sys.dC(s, i, j, k-1) * np.sqrt(k/2) * sys.species[s].scale[2]
        terms += sys.dC(s, i, j, k+1) * np.sqrt((k+1)/2) * sys.species[s].scale[2]
        return terms

    @staticmethod
    def cylindrical(sys, s, i, j, k):
        raise NotImplementedError("Cylindrical advection operator not implemented!")

class Acceleration():
    @staticmethod
    def electric(sys, s, i, j, k, E):
        terms = - E * sys.species[s].q/sys.species[s].m * np.sqrt(2*k) / sys.species[s].scale[2] * sys.C(s, i, j, k-1)
        return terms

    @staticmethod
    def magnetic(sys, s, i, j, k, B):
        raise NotImplementedError("Magnetic field acceleration operator not implemented!")

    @staticmethod
    def centripetal(sys, s, i, j, k):
        raise NotImplementedError("Centripetal acceleration operator not implemented!")

    @staticmethod
    def coriolis(sys, s, i, j, k):
        raise NotImplementedError("Coriolis acceleration operator not implemented!")

class KineticEquation(): # Implementation of kinetic equation in form dC/dt = terms
    @staticmethod
    def cartesian(sys, s, i, j, k, E=0, B=0):
        terms = 0
        terms -= Advection.cartesian(sys, s, i, j, k)
        terms -= Acceleration.electric(sys, s, i, j, k, E)
        terms += Collision.lenard_bernstein(sys, s, i, j, k)
        # print(terms)
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