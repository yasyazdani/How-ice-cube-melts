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
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.colors import LinearSegmentedColormap
from dedalus.extras.plot_tools import plot_bot_2d

# Initial Condition Constants
u_0x = 0
u_0z = 0
v_0 = 0
p_0 = 0
T_0 = 0 
C_0 = 1
ft_0 = 0

# Dimensional parameters
UU = 0    
T_B = 20  # C  
c_p = 4.2  # J/g*C
L_T = 3.34*(10**2)  # J/g
C_B = 2  # g/kg
nu = 1.3*(10**-1)   # cm**2/s; physical viscosity is 10 times smaller.
kappa = 1.3 *(10**-2) # cm**2/s; NEVER USED (physical value 10 times smaller)
mu = 1.3*(10**-3)   # cm**2/s; NEVER USED (physical value 10 times smaller)
em = 0.056  # C/(g/kg)
LL, HH = 10, 5  # cm
ll, hh = 2, 2  # cm
epsilon = 0.04   # cm
# Parameters
Lx, Lz = LL,HH
Nx, Nz = 256, 128
dealias = 3/2

stop_sim_time = 2 #20   

timestepper = d3.RK222
timestep = 1e-4
max_timestep = 1e-4  # will need to adjust to constant
dtype = np.float64


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


# new                            
ww = 7.2921 * (10**-5) # angular velocity      change the value later
phi = np.deg2rad(43.7) #  latitude.           change it later
f_c = 2*ww* np.sin(phi) # Coriolis parameter
f_c_r = 2*ww* np.cos(phi) # reciprocal Coriolis parameter

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)


xbasis = d3.RealFourier(coords['x'], size=256, bounds=(0, Lx), dealias=dealias)
zbasis = d3.Chebyshev(coords['z'], size=128, bounds=(0, Lz), dealias=dealias) # check if its not 2

# Fields
p = dist.Field(name='p', bases=(xbasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
v = dist.Field(name='v', bases=(xbasis, zbasis))
T = dist.Field(name='T', bases=(xbasis,zbasis))
C = dist.Field(name='C', bases=(zbasis, xbasis)) 
Phase = dist.Field(name='Phase', bases=(xbasis, zbasis))
Phase_t = dist.Field(name='Phase_t', bases=(xbasis, zbasis)) # dt(Phase) = Phase_t

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
x = xbasis.local_grid(dist=dist, scale=1)
z = zbasis.local_grid(dist=dist, scale=1)
local_x = dist.local_grid(xbasis)
local_z = dist.local_grid(zbasis)
local_kx = dist.local_modes(xbasis)
local_kz = dist.local_modes(zbasis)
kx = local_kx
kz = local_kz


# new
par = dist.Field(name='par', bases=(xbasis, zbasis), dtype=np.float64) # to control Buoyancy
par['g'] = np.tanh(-(z-HH)/.05)*np.tanh(z/.05)
par['c'] *= np.exp(-kx**2/5e6)  # spectral smoothing


# Functions  
def sigmoid(x, a=1):
    return 0.5*(np.tanh(x/a)+1)

# new
wall = dist.Field(name='wall', bases=(xbasis, zbasis), dtype=np.float64) 
wall['g'] = 0  # no wall 
wall['c'] *= np.exp(-kx**2/5e6)  # spectral smoothing

domain = Domain(dist, (xbasis, zbasis),)   


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


# #new
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

buoyancy = forcing(C,T)


# expression for Phase
Phase_expr = ( sigmoid(z - (HH - hh), a=2 * epsilon) *
    sigmoid(x - (LL - ll) / 2, a=2 * epsilon) *
    sigmoid(-(x - (LL + ll) / 2), a=2 * epsilon))


# Problem

problem = d3.IVP([p,b,C, T, u, v,Phase,Phase_t, tau_p, tau_T1, tau_T2, tau_u1, tau_u2,
                 tau_v1, tau_v2, tau_C1, tau_C2,tau_Phase1, tau_Phase2], namespace=locals())
problem.add_equation("b = buoyancy * par  ")  
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - (1/(Pr*Re)) * div(grad_T) + lift(tau_T2) - SS*dt(Phase) = -(1-Phase) * (u@grad(T) + T * (u@grad(Phase))) - (wall/eta)*(T-1)") 
problem.add_equation("dt(C) - (1/(Sc*Re))*div(grad_C) + lift(tau_C2) = -(1-Phase)* u@grad(C) + (C*Phase_t - grad_C@grad_Phase)/((Sc*Re)*(1-Phase+delta))- (wall/eta)*(C-1)") 
problem.add_equation(" dt(u) + grad(p)  - (1/Re)*div(grad_u) + lift(tau_u2) - f_c*(v*ex) = b*ez  - u@grad(u) - (Phase/eta)*u - (wall/eta)*u") 
problem.add_equation("(AA/epsilon)*dt(Phase) - (GG/epsilon)*div(grad_Phase) + lift(tau_Phase2) = - (1/epsilon**2)*Phase*(1-Phase)*((GG/epsilon)*(1-2*Phase) + (T+MM*C))")
problem.add_equation("dt(v) + f_c*(u@ex) + - (1/Re)*div(grad_v) + lift(tau_v2) = - u@grad(v)- (wall/eta)*v")
problem.add_equation("dt(Phase) - Phase_t = 0 ")
problem.add_equation("(u@ez)(z=0) = 0")
problem.add_equation("dz(u@ex)(z=0) = 0")
problem.add_equation("dz(u@ex)(z=Lz) = 0")
problem.add_equation("(u@ez)(z=Lz) = 0")
problem.add_equation("dz(T)(z=Lz) = 0")
problem.add_equation("dz(T)(z=0) = 0")
problem.add_equation("dz(C)(z=Lz) = 0")
problem.add_equation("dz(C)(z=0) = 0")
problem.add_equation("dz(v)(z=0) = 0")
problem.add_equation("dz(v)(z=Lz) = 0")
problem.add_equation("integ(p) = 0")  # Pressure gauge
problem.add_equation("dz(Phase)(z=Lz) = 0") 
problem.add_equation("dz(Phase)(z=0) = 0") 



# Initial conditions
u['g'][0] = u_0x 
u['g'][1] = u_0z 
Phase['g'] = Phase_expr
v['g'] = v_0
C['g'] = C_0
p['g'] = p_0  #pressure fluctuation
T['g'] = 1-Phase['g']  
b['g'] = -9.8 * 100 * (sw.dens0(C_B * C['g'], T_B * T['g']) - rho0) / rho0



# # Solver
solver = problem.build_solver(timestepper)
solver.evaluate_handlers_now(dt=0)
solver.stop_sim_time = stop_sim_time

# Analysis
# 2D fields
snapshots = solver.evaluator.add_file_handler(
    'snapshots', sim_dt=0.25, max_writes=50)  

task_names = ['buoyancy', 'Temperature', 'Concentration', 'Phase', 'Phase_t', 'spanwise vorticity', 'vertical vorticity','x velocity','z velocity','y velocity']
tasks = [b, T, C, Phase, Phase_t, -d3.div(d3.skew(u)), dx(v), u@ex , u@ez, v ]

for task, name in zip(tasks, task_names):
    if name not in snapshots.tasks:
        snapshots.add_task(task, name=name)

# 1D vertical profiles
profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.25)
profiles.add_task(d3.Average(b, 'x'), name='b')  
profiles.add_task(d3.Average(T, 'x'), name='T') 
profiles.add_task(d3.Average(C, 'x'), name='C')  
profiles.add_task(d3.Average(Phase, 'x'), name='Phase')   
profiles.add_task(d3.Average(Phase_t, 'x'), name='Phase_t')   
profiles.add_task(d3.Average(u@ex, 'x'), name='u')
profiles.add_task(d3.Average(v, 'x'), name='v')

# 0D (time series)
timeseries = solver.evaluator.add_file_handler('timeseries', sim_dt=0.25)

task_names = ['KE', 'HF', 'CF', 'HCF', 'energy', 'salt', 'volume', 'wb', 'GBP']
tasks = [
    d3.Average(0.5*(u@u + v**2) * rho0, ('x', 'z')),
    d3.Average((u@ez)*T, ('x', 'z')),
    d3.Average((u@ez)*C, ('x', 'z')),
    d3.Average(C*(u@ex), ('x', 'z')),
    d3.Integrate((T - SS*Phase),('x','z')),
    d3.Integrate((1-Phase)*C,('x','z')),
    d3.Integrate(Phase, ('x', 'z')),
    d3.Average((u@ez)*b, ('x', 'z')),
    d3.Average(b*(u@ex), ('x', 'z'))
]

for task, name in zip(tasks, task_names):
    if name not in timeseries.tasks:
        timeseries.add_task(task, name=name)


# CFL
# CFL condition and timestep adjustment
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5,
             threshold=0.05, max_change=1.5, min_change=0.5,
             max_dt=max_timestep)
CFL.add_velocity(u)


# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((np.sqrt(u@u + v**2))*Re, name='Re') 


# Add these before the main loop
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        # timestep = dt #CFL.compute_timestep()
        timestep = CFL.compute_timestep()
        solver.step(timestep)
      

        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %
                        (solver.iteration, solver.sim_time, timestep, max_Re))
            # logger.info('Field b: %s', b['g']) 
            # logger.info('Field u: %s', u['g'])
            # logger.info('Field T: %s', T['g']) 
            logger.info('Field C: %s', C['g']) 
            # logger.info('Field Phase: %s', Phase['g'])
      

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()



