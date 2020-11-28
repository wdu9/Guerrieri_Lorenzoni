# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:30:31 2020

@author: wdu
"""
import numpy as np 
from builtins import str
from builtins import range
from builtins import object
from HARK import AgentType, NullFunc, HARKobject, makeOnePeriodOOSolver
from scipy.io import loadmat
from HARK.interpolation import  LinearInterp
from copy import copy, deepcopy
from HARK.utilities import getArgNames, NullFunc


#def makeOnePeriodOOSolver(solver_class):
    #def onePeriodSolver(**kwds):
        #solver = solver_class(**kwds)
        # not ideal; better if this is defined in all Solver classes
        #if hasattr(solver, "prepareToSolve"):
            #solver.prepareToSolve()
            #solution_now = solver.solve()
        #return solution_now

        #onePeriodSolver.solver_class = solver_class
    # This can be revisited once it is possible to export parameters
        #onePeriodSolver.solver_args = getArgNames(solver_class.__init__)[1:]

        #return onePeriodSolver
    


class GLConsumerSolution(HARKobject):
    """
    A class representing the solution of a single period of a GL
    problem.  The solution includes a the consumption policy from the next period 
    and an array whose elements are LinearInterp Objects.
    """

    distance_criteria = ["vPfunc"]

    def __init__(
        self,
        Cpol = None,
        cFuncs = None,
       
    ):
        """
        The constructor for a new GLConsumerSolution object.
        Parameters
        ----------
        Cpol : Array
            Consumption Policy from next period. 13 by 200 array 
        Cfuncs : Array
            13 by 1 Array whose elements are objects of the 'LinearInterp' Class. 
            The ith element is the linearinterpolation between the Bond Grid and consumption for a certain level of productivity.
           
        Returns
        -------
        None
        """
        # Change any missing function inputs to NullFunc
        self.Cpol = Cpol if Cpol is not None else NullFunc()
        self.cFuncs = cFuncs if cFuncs is not None else NullFunc()

    
            
            
#------------------------------------------------------------------------------
            

# Calibration targets
NE       = 0.4;     # avg hours of employed
nu_Y     = 0.4;     # UI benefit per GDP
B_4Y     = 1.6;     # liquid wealth per annual GDP
D1_4Y    = 0.18;    # HH debt to annual GDP in initial ss
D2_4Y    = 0.08;    # HH debt to annual GDP in terminal ss

#fixed params
Rfree=1.00625
frisch = 1      # avg Frisch elast.
CRRA=4
DiscFac  = 0.8**(1/4);              # discount factor

eta    = (1/frisch) * (1 - NE) / NE


pssi = NE**(-CRRA) * (1-NE)**eta; # disutility from labor as if representative agent


gameta = CRRA / eta;

# Initial guesses for calibrated parameters, NE ~ Y
DiscFac  = 0.8**(1/4);              # discount factor
nu   = nu_Y  * NE;             # UI benefits
B    = B_4Y  * NE * 4;         # net supply of bonds
phi = D1_4Y * NE * 2;         # borrowing constraint in initial ss
phi2 = D2_4Y * NE * 2;         # borrowing constraint in terminal ss



#------------------------------------------------------------------------------
class GLsolver(HARKobject):
    #Pr is transition matrix for Markov chain, 
    #C_next is the grid of consumption points from next period

    def __init__(
        self,
        solution_next,
        DiscFac,
        CRRA,
        Rfree,
        eta,
        nu,
        pssi,
        phi,
        B,
        
    ):
        self.assignParameters(
            solution_next=solution_next,
            CRRA=CRRA,
            Rfree=Rfree,
            DiscFac=DiscFac,
            phi=phi,
            eta=eta,
            nu=nu,
            pssi=pssi,
            B=B,
        
        )
    def mkCpol(self):    
        #load the income process 
        Matlabdict = loadmat('inc_process.mat')
        data = list(Matlabdict.items())
        data_array=np.asarray(data)
        x=data_array[3,1]
        Pr=data_array[4,1]
        pr = data_array[5,1]
    
        theta = np.concatenate((np.array([1e-10]).reshape(1,1),np.exp(x).reshape(1,12)),axis=1).reshape(13,1)
        fin   = 0.8820    #job-finding probability
        sep   = 0.0573    #separation probability
        cmin= 1e-6  # lower bound on consumption 
        
        
        
        #constructing transition Matrix
        G=np.array([1-fin]).reshape(1,1)
        A = np.concatenate((G, fin*pr), axis=1)
        B= sep**np.ones(12).reshape(12,1)
        D=np.concatenate((B,np.multiply((1-sep),Pr)),axis=1)
        Pr = np.concatenate((A,D))
        
        
        # find new invariate distribution
        pr = np.concatenate([np.array([0]).reshape(1,1), pr],axis=1)
        
        dif = 1
        while dif > 1e-5:
            pri = pr.dot(Pr)
            dif = np.amax(np.absolute(pri-pr))
            pr  = pri
        
            
        fac = ((pssi / theta)** (1/eta)).reshape(13,1)  
            
        tau = (nu*pr[0,0] + (Rfree-1)/(Rfree)*B) / (1 - pr[0,0]) # labor tax
            
        z  = np.insert(-tau*np.ones(12),0,nu).T
                 
                 
                 
            
            #this needs to be specified differently to incorporate choice of phi and interest rate
        Matlabcl=loadmat('cl')
        cldata=list(Matlabcl.items())
        cldata=np.array(cldata)
        cl=cldata[3,1].reshape(13,1)
        # print(cl)
        Matlabgrid=loadmat('Bgrid')
        griddata=list(Matlabgrid.items())
        datagrid_array=np.array(griddata)
        print(datagrid_array[3,1])
        Bgrid_uc=datagrid_array[3,1]
    
        #setup grid based on constraint phi
        Bgrid=[]
        for i in range(200):
            if Bgrid_uc[0,i] > self.phi:
                Bgrid.append(Bgrid_uc[0,i])
        Bgrid = np.array(Bgrid).reshape(1,len(Bgrid))
        
        
        Cnext = self.solution_next.Cpol
            
    
        phi= D1_4Y * NE * 2
        expUtil= np.dot(Pr,(Cnext**(-CRRA)))
        print(expUtil)
        Cnow=[]
        Nnow=[]
        Bnow=[]
        self.Cnowpol=[]
            
            #unconstrained
        for i in range(13):
            Cnow.append ( np.array(((Rfree) * DiscFac) * (expUtil[i] **(-1/CRRA)))) #euler equation
            Nnow.append(np.array(np.maximum((1 - fac[i]*((Cnow[i])**(CRRA / eta))),0))) #labor supply FOC
            Bnow.append(np.array(Bgrid[0] / (Rfree) + Cnow[i] - theta[i]**Nnow[i] - z[i])) #Budget Constraint
                
            #constrained, constructing c pts between -phi and Bnow[i][0]  
            if Bnow[i][0] > -phi:
                c_c = np.linspace(cl[i,0], Cnow[i][0], 100)
                n_c = np.maximum(1 - fac[i]*(c_c**gameta),0)  # labor supply
                b_c = -phi/Rfree + c_c - theta[i]**n_c - z[i] # budget
                Bnow[i] = np.concatenate( [b_c[0:98], Bnow[i]])
                Cnow[i] = np.concatenate([c_c[0:98], Cnow[i]])
                
                    
            self.Cnowpol.append( LinearInterp(Bnow[i], Cnow[i]))
                
        self.Cnowpol=np.array(self.Cnowpol).reshape(13,1)
        self.Cpol=[]
        for i in range(13):
            self.Cpol.append(self.Cnowpol[i,0].y_list)
        self.Cpol = np.array(self.Cpol)
        print(Cpol)
        
    
    def solve(self):
        """
        Solves the one period problem.
        Parameters
        ----------
        None
        Returns
        -------
        solution : GLConsumerSolution
            The solution to this period's problem.
        """
        self.mkCpol()
        solution = GLConsumerSolution(
            cFuncs=self.CnowPol,
            Cpol=self.Cpol,
            
          
        )
        return solution

#------------------------------------------------------------------------------
    
# Make a dictionary to specify a GL consumer type
init_GL = {
    'CRRA': 4.0,          # Coefficient of relative risk aversion,
    'Rfree': 1.00625,     # Interest factor on assets
    'DiscFac': 0.8**(1/4),# Intertemporal discount factor
    'phi': 1.60054,       # Artificial borrowing constraint
    'eta': 1.5,
    'nu': .16000,
    'pssi': 18.154609,
    'B': 2.56,
    #'AgentCount': 10000,  # Number of agents of this type (only matters for simulation)
    #'aNrmInitMean' : 0.0, # Mean of log initial assets (only matters for simulation)
    #'aNrmInitStd' : 1.0,  # Standard deviation of log initial assets (only for simulation)
    #'pLvlInitMean' : 0.0, # Mean of log initial permanent income (only matters for simulation)
    #'pLvlInitStd' : 0.0,  # Standard deviation of log initial permanent income (only matters for simulation)
    #'PermGroFacAgg' : 1.0,# Aggregate permanent income growth factor: portion of PermGroFac attributable to aggregate productivity growth (only matters for simulation)
    #'T_age' : None,       # Age after which simulated agents are automatically killed
    #'T_cycle' : 1         # Number of periods in the cycle for this agent type
}
    
class GLconsType(AgentType):
   
    
    

    def __init__(self, cycles=0, verbose=1, quiet=False, **kwds):
        self.time_vary = []
        self.time_inv = ["CRRA", "Rfree", "DiscFac", "phi","eta","nu","pssi","B"]
        self.state_vars = []
        self.shock_vars = []
        cmin= 1e-6  # lower bound on consumption
        
         

        Matlabgrid=loadmat('Bgrid')
        griddata=list(Matlabgrid.items())
        datagrid_array=np.array(griddata)
        print(datagrid_array[3,1])
        Bgrid_uc=datagrid_array[3,1]
    
        params = init_GL.copy()
        #params.update(kwds)
        kwds = params
        print(kwds)
        #setup grid based on constraint phi
        Bgrid=[]
        for i in range(200):
            if Bgrid_uc[0,i] > phi:
                Bgrid.append(Bgrid_uc[0,i])
        Bgrid = np.array(Bgrid).reshape(1,len(Bgrid))

    #initial Guess for Cpolicy
        Cguess = np.maximum(Rfree*np.ones(13).reshape(13,1).dot(Bgrid),cmin)
        # Define some universal values for all consumer types
         
        solution_terminal_ = GLConsumerSolution(
                Cpol=Cguess,
                )
        
        AgentType.__init__(
        self,
            solution_terminal=deepcopy(solution_terminal_),
            cycles=cycles,
            pseudo_terminal=False,
            **kwds
        ) #going to need to specify **kwds and solution_terminal_
       
        self.verbose = verbose
        self.quiet = quiet
        self.solveOnePeriod = makeOnePeriodOOSolver(GLsolver)
        #set_verbosity_level((4 - verbose) * 10)
        
        
#-------------------------------------------------------------------------------
        
example = GLconsType(**init_GL)
example.solve()

        
   