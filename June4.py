
import numpy as np
import dedalus.public as d3
from dedalus.core.domain import *  # new
import logging
logger = logging.getLogger(__name__)
import seawater as sw
from dedalus.core.operators import GeneralFunction
# from dedalus.tools import post
from os.path import join
# from dedalus.extras import flow_tools
import time
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Initial Condition Constants
u_0 = 2
w_0 = 0
p_0 = 5
T_0 = 0 
C_0 = 5
ft_0 = 2

# Dimensional parameters
UU = 6
T_B = 10  # C  # it was 20
c_p = 4.2  # J/g*C
L_T = 3.34*(10**2)  # J/g
C_B = 30  # g/kg
nu = 1.3*(10**-1)   # cm**2/s; physical viscosity is 10 times smaller.
kappa = 1.3 *(10**-2) # cm**2/s; NEVER USED (physical value 10 times smaller)
mu = 1.3*(10**-3)   # cm**2/s; NEVER USED (physical value 10 times smaller)
em = 0.056  # C/(g/kg)
LL, HH = 10, 5  # cm
ll, hh = 2, 2  # cm
epsilon = 4*(10**-2)   # cm



# new
ww = 7.2921 * (10**-5) # angular velocity      change the value later
phi = np.deg2rad(43.7) #  latitude.           change it later
f_c = 2*ww* np.sin(phi) # Coriolis parameter

f_c_r = 2*ww* np.cos(phi) # reciprocal Coriolis parameter


# Non-dimensional parameters
Pr = 7
Sc = 50/4
delta = 1*(10**-4) 
beta = 4/2.648228  
Re = 1 / nu
SS = L_T / (c_p * T_B)
MM = (em * C_B * C_0) #/ T_B
AA = epsilon * (SS * Re * Pr) * (5 / 6)
AA = np.float64(AA)
GG = epsilon
GG = np.float64(GG)
eta = 10**-1 * Re * (beta * epsilon) ** 2  # not "optimal"
eta = np.float64(eta) 
rho0 = sw.dens0(s=C_B * C_0, t=T_B)
# rho0 = sw.dens0(s=5, t=20)
# Parameters
Lx, Lz = 2, 1
Nx, Nz = 24, 12
dealias = 3/2
stop_sim_time = 2 #20
timestepper = d3.RK222
max_timestep = 1e-1  # will need to adjust to constant
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz, 0), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
v = dist.Field(name='v', bases=(xbasis, zbasis))

T = dist.Field(name='T', bases=(xbasis,zbasis))
C = dist.Field(name='C', bases=(zbasis, xbasis)) 
Phase = dist.Field(name='Phase', bases=(zbasis, xbasis))
Phase_t = dist.Field(name='Phase_t', bases=(zbasis, xbasis)) # dt(Phase) = Phase_t


tau_p = dist.Field(name='tau_p')
tau_Phase1 = dist.Field(name='tau_Phase1', bases=xbasis)
tau_Phase2 = dist.Field(name='tau_Phase2', bases=xbasis)
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)
tau_C1 = dist.Field(name='tau_C1', bases=xbasis)
tau_C2 = dist.Field(name='tau_C2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
tau_v1 = dist.Field(name='tau_v1', bases=xbasis)
tau_v2 = dist.Field(name='tau_v2', bases=xbasis)


# new
local_x = dist.local_grid(xbasis)
local_z = dist.local_grid(zbasis)
local_kx = dist.local_modes(xbasis)
local_kz = dist.local_modes(zbasis)
x = local_x
z = local_z
kx = local_kx
kz = local_kz

# new
par = dist.Field(name='par', bases=(xbasis, zbasis), dtype=np.float64) 
par['g'] = np.tanh(-(z-HH)/.05)*np.tanh(z/.05)
par['c'] *= np.exp(-kx**2/5e6)  # spectral smoothing


# Substitutions
# kappa = (Rayleigh * Prandtl)**(-1/2) # I dont need them?
# nu = (Rayleigh / Prandtl)**(-1/2)  # I dont need them?


x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
def lift(A): return d3.Lift(A, lift_basis, -1)
def dx(A): return d3.Differentiate(A, coords['x'])
def dz(A): return d3.Differentiate(A, coords['z'])


grad_u = d3.grad(u) + ez*lift(tau_u1)  # First-order reduction
grad_Phase = d3.grad(Phase) + ez*lift(tau_Phase1)  # First-order reduction
grad_v = d3.grad(v) + ez*lift(tau_v1)  # First-order reduction
grad_T = d3.grad(T) + ez*lift(tau_T1)  # First-order reduction
grad_C = d3.grad(C) + ez*lift(tau_C1)  # First-order reduction


domain = Domain(dist, (xbasis, zbasis))   # check the domain later



# new
def sigmoid(x, a=1):
    return 0.5*(np.tanh(x/a)+1)

#new
def compute_buoyancy(C,T):
    buoyancy_array = -9.8 * 100 * (sw.dens0(C_B * C, T_B * T) - rho0) / rho0
    return  buoyancy_array
def forcing_function(*args):
    C=args[0]['g']  
    T=args[1]['g'] 
    f = compute_buoyancy(C,T)
    return f


def forcing(*args, domain= domain ,F=forcing_function):
    return d3.GeneralFunction(dist= dist,domain=domain,layout='g',tensorsig=(),dtype=np.float64,func=F,args=args)



# # Set up the Phase field
# Phase['g'] = 0   # Set all to 0 (Water)
# # Set the ice block 
# ice_x_start = 20 #Nx // 2 - 2
# ice_x_end = 24 #Nx // 2 + 2
# ice_z_start = 5 #Nz//2 -2
# ice_z_end = 7 #Nz//2 +2
# # Phase['g'][ice_x_start:ice_x_end,ice_z_start:ice_z_end ] = 1  # Set ice to 1







# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"

problem = d3.IVP([p,b,C, T, u, v,Phase,Phase_t, tau_p, tau_T1, tau_T2, tau_u1, tau_u2,
                 tau_v1, tau_v2, tau_C1, tau_C2,tau_Phase1, tau_Phase2], namespace=locals())
problem.add_equation("b = forcing(C,T)  ")  # I need to keep it the way it is otherwise-> ValueError: Non-square system
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - (1/(Pr*Re)) * div(grad_T) + lift(tau_T2) - SS*dt(Phase) = -(1-Phase) * (u@grad(T) + T * (u@grad(Phase)))") 
problem.add_equation("dt(C) - (1/(Sc*Re))*div(grad_C) + lift(tau_C2) = -(1-Phase)* u@grad(C) + (C*Phase_t - grad_C@grad_Phase)/((Sc*Re)*(1-Phase+delta))") 
problem.add_equation(" dt(u) + grad(p)  - (1/Re)*div(grad_u) + lift(tau_u2) - f_c*(v*ex) = par*b*ez  - u@grad(u) - (Phase/eta)*u ") 
problem.add_equation("(AA/epsilon)*dt(Phase) - (GG/epsilon)*div(grad_Phase) + lift(tau_Phase2) = - (1/epsilon**2)*Phase*(1-Phase)*((GG/epsilon)*(1-2*Phase) + (T+MM*C))")
problem.add_equation("dt(v) + f_c*(u@ex) + - (1/Re)*div(grad_v) + lift(tau_v2) = - u@grad(v)")
problem.add_equation("dt(Phase) - Phase_t = 0 ")
problem.add_equation("dz(T)(z=-Lz) = 0")
problem.add_equation("dz(C)(z=-Lz) = 0")
problem.add_equation("dz(u@ex)(z=0) = 0")
problem.add_equation("(u@ez)(z=0) = 0")
problem.add_equation("dz(v)(z=0) = 0")
problem.add_equation("dz(u@ex)(z=-Lz) = 0")
problem.add_equation("(u@ez)(z=-Lz) = 0")
problem.add_equation("dz(v)(z=-Lz) = 0")
problem.add_equation("dz(T)(z=0) = 0")
problem.add_equation("dz(C)(z=0) = 0")
problem.add_equation("integ(p) = 0")  # Pressure gauge
problem.add_equation("dz(Phase)(z=-Lz) = 0") #?
problem.add_equation("dz(Phase)(z=0) = 0") #?

# problem.add_equation("par(z=0)= - par(z= -Lz)") # new it shoubd be odd in z, but cant add the conditions
# problem.add_equation("par(z=-Lz/2)= 0") # new


# Initial conditions (it was for b) change the inital condition later
# T.fill_random('g', seed=42, distribution='normal', scale=1e-3)  # Random noise
# T['g'] *= z * (-Lz - z)  # Damp noise at walls

# # Initial conditions 
b['g'] = -9.8 * 100 * (sw.dens0(C_B * C['g'], T_B * T['g']) - rho0) / rho0
u['g'] = u_0 
p['g'] = p_0
T['g'] = 1-Phase['g']   ##### why?????????
C['g'] = C_0
        # C['g'] = well(z-(HH-hh), x-(LL-ll)/2, -(x-(LL+ll)/2))
        # C['g'] = abs(1-f['g']) #TZ 20240122: uncomment this to make salt only exist in liquid
p['g'] = p_0  #pressure fluctuation
v['g'] = 0
Phase['g'] = (sigmoid(z-(HH-hh), a=2*epsilon) * sigmoid(x-(LL-ll)/2, a=2*epsilon) * sigmoid(-(x-(LL+ll)/2), a=2*epsilon))



# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time



# Analysis
# 2D fields
snapshots = solver.evaluator.add_file_handler(
    'snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(T, name='Temperature') #new
# snapshots.add_task(C, name='Concentration') #new

snapshots.add_task(-d3.div(d3.skew(u)), name='spanwise vorticity')
snapshots.add_task(dx(v), name='vertical vorticity')
# snapshots.add_task(1 + R - J, name='PV')  # eventually

# 1D vertical profiles
profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.25)
snapshots.add_task(d3.Average(b, 'x'), name='b')
snapshots.add_task(d3.Average(T, 'x'), name='T') #new
snapshots.add_task(d3.Average(C, 'x'), name='C')  # new

snapshots.add_task(d3.Average(u@ex, 'x'), name='u')
snapshots.add_task(d3.Average(v, 'x'), name='v')


# 0D (time series)
timeseries = solver.evaluator.add_file_handler('timeseries', sim_dt=0.25)
timeseries.add_task(d3.Average(0.5*(u@u + v**2), ('x', 'z')), layout='g',
                    name='KE')   # 0.5 * (u@u + v**2) * rho0 ? or without rho0 ? check the dimention later
timeseries.add_task(d3.Average((u@ez)*T, ('x', 'z')), layout='g',
                    name='uz_T')
timeseries.add_task(d3.Average(T*(u@ex), ('x', 'z')), layout='g',
                    name='ux_T')
timeseries.add_task(d3.Average((u@ez)*C, ('x', 'z')), layout='g',
                    name='uz_C')
timeseries.add_task(d3.Average(C*(u@ex), ('x', 'z')), layout='g',
                    name='ux_C')
timeseries.add_task(d3.Integrate((T - SS*Phase),('x','z')),layout='g', name='energy')

timeseries.add_task(d3.Integrate((1-Phase)*C,('x','z')), layout='g', name='salt')

timeseries.add_task(d3.Integrate(Phase, ('x', 'z')), layout='g', name='volume')
# "q"  #add later
# "buoyancy" #add later

# not sure if I need them
# timeseries.add_task(d3.Average((u@ez)*b, ('x', 'z')), layout='g',          
#                     name='wb')
# timeseries.add_task(d3.Average(b*(u@ex), ('x', 'z')), layout='g',
                    # name='GBP')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5,
             threshold=0.05, max_change=1.5, min_change=0.5,
             max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((np.sqrt(u@u + v**2))* Re, name='Re')


# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        # Print values to check updates
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %
                        (solver.iteration, solver.sim_time, timestep, flow.max('Re')))
            logger.info('Field b: %s', b['g']) # new
            logger.info('Field u: %s', u['g']) # new
            logger.info('Field T: %s', T['g']) # new
            logger.info('Field C: %s', C['g']) # new
            logger.info('Field Phase: %s', Phase['g']) # new

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()


