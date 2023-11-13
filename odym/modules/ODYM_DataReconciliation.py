# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:43:26 2023

@author: fabio carrer, fabio.carrer@ntnu.no

ODYM extension to perform Data Reconciliation.
Repository for ODYM classes, documentation, and tutorials: https://github.com/IndEcol/ODYM

"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import sympy as sym
import sys
from tqdm import tqdm
from sympy.solvers import solve


def DataReconciliation(MFASystem, ParametersEquationsList, InitialEstimateDict, tol=1e-6):
        """ 
        Method that performs non-linear data reconciliation under the assumption of Normal distributions of uncertainty
        based on https://doi.org/10.1016/j.serj.2016.06.002.
        
        Parameters
        -----------
            MFASystem (MFAsystem): MFAsystem objects from ODYM_Classes library.
            ParametersEquationsList (list): List of model equations involving parameters.
            InitialEstimateDict (dict): Dictionary of initial estimates values.
            tol (float): Convergence tolerance. Default: 1e-6.
            
        Returns
        --------
            results (Dataframe): A table that summarises the results from data reconciliation.
            MassBalance (list): List of Mass-Balance computed for each process of the system.
            
        """
        
        for flow in MFASystem.FlowDict:
            try:
                if len(MFASystem.FlowDict[flow].Indices) > 1:
                    sys.exit(f'Flow {flow} has more than one dimension. This method cannot perform reconciliantion on multi-layer systems.' )
            except:
                pass
            
        MassBalanceEquationsDict, EquationsDict = {},{}
        
        for key in MFASystem.ProcessList:
            if key.ID !=0:
                MassBalanceEquationsDict[str(key.ID)] = ''
            
        for key in MFASystem.FlowDict:
            if MFASystem.FlowDict[key].P_Start == None:
                MassBalanceEquationsDict[str(MFASystem.FlowDict[key].P_End)] += '-'+str(key)
            else:
                if MFASystem.FlowDict[key].P_Start != 0:
                    MassBalanceEquationsDict[str(MFASystem.FlowDict[key].P_Start)] += '-'+str(key)
                if MFASystem.FlowDict[key].P_End != 0:
                    MassBalanceEquationsDict[str(MFASystem.FlowDict[key].P_End)]   += '+'+str(key)
            
        for key in MassBalanceEquationsDict.keys():
            EquationsDict[key] = MassBalanceEquationsDict[key]
        
        for equation in ParametersEquationsList:
            EquationsDict[list(MFASystem.ParameterDict.keys())[ParametersEquationsList.index(equation)]] = equation.split('=')[0]
        
        y, x, x_hat, y_hat, q, z  = [],[],[],[],[],{} 
        for key in MFASystem.FlowDict:
            if MFASystem.FlowDict[key].Values != None and MFASystem.FlowDict[key].Uncert == None: 
                z[key]=MFASystem.FlowDict[key].Values
            elif MFASystem.FlowDict[key].Values == None:  
                y.append(key)
            else:  
                x.append(key)
                q.append((MFASystem.FlowDict[key].Values*MFASystem.FlowDict[key].Uncert)**2)
                x_hat.append(MFASystem.FlowDict[key].Values)
        for key in MFASystem.ParameterDict:
            if MFASystem.ParameterDict[key].Values != None and MFASystem.ParameterDict[key].Uncert == None: 
                z[key]=MFASystem.ParameterDict[key].Values
            elif MFASystem.ParameterDict[key].Values == None:  
                y.append(key)
            else:
                x.append(key)
                q.append((MFASystem.ParameterDict[key].Values*MFASystem.ParameterDict[key].Uncert)**2)
                x_hat.append(MFASystem.ParameterDict[key].Values)
        x_tilde = x_hat 
        Q = np.diag(q)
        
        for key in y:
            y_hat.append(InitialEstimateDict[key])
       
        J_y = [[None for i in range(len(y))] for j in range(len(EquationsDict))]
        for i in range(len(EquationsDict)):
            for j in range(len(y)): 
                J_y[i][j] = sym.diff(list(EquationsDict.values())[i],y[j])
        J_x = [[None for i in range(len(x))] for j in range(len(EquationsDict))]
        
        J_x_hat = [[None for i in range(len(x))] for j in range(len(EquationsDict))]
        for i in range(len(EquationsDict)):
            for j in range(len(x)): 
                J_x[i][j] = sym.diff(list(EquationsDict.values())[i],x[j])
        J_y_hat = [[None for i in range(len(y))] for j in range(len(EquationsDict))]
        for i in range(len(EquationsDict)):
            for j in range(len(y)): 
                J_y[i][j] = sym.diff(list(EquationsDict.values())[i],y[j])
        
        subs = [ (y[i],y_hat[i]) for i in range(len(y))]
        subs.extend( [ (x[i],x_hat[i]) for i in range(len(x))] )
        subs.extend( [ (key, z[key]) for key in list(z.keys())] )
        for i in range(len(EquationsDict)):
            for j in range(len(x)):
                J_x_hat[i][j] = J_x[i][j].subs(subs)
        for i in range(len(EquationsDict)):
            for j in range(len(y)):
                J_y_hat[i][j] = J_y[i][j].subs(subs)
        
        EquationsList = list(EquationsDict.values())
        f_hat = [sym.sympify((EquationsList[i])).subs(subs) for i in range(len(EquationsList))]
       
        A = np.hstack( (np.array(J_y_hat),np.array(J_x_hat),np.array(f_hat)[:,None])  ) 
        A_rref, Acy,Acx,Acz,Ary,Arx,Arz = get_RREF_blocks(A,len(y),len(x))
        
        x_tilde = np.array(x_tilde).reshape(-1,1)
        x_hat = np.array(x_hat).reshape(-1,1)
        y_hat = np.array(y_hat).reshape(-1,1)
        
        variables_to_convert = [Q, x_hat, x_tilde, Acy, Acx, Acz, Ary, Arx, Arz, y_hat]
        variables_converted = [np.array(var, dtype=float) for var in variables_to_convert]
        Q, x_hat, x_tilde, Acy, Acx, Acz, Ary, Arx, Arz, y_hat = variables_converted
        
        x_new = np.subtract(x_tilde ,  Q@np.transpose(Arx)@np.linalg.inv(Arx@Q@np.transpose(Arx))@(np.add(Arx@(x_tilde-x_hat),Arz)))
        
        rows_to_remove, columns_to_remove = [], []
        for i in range(len(Acy)):
            if (Acy[i,:]!=0).sum() !=1:
                rows_to_remove.append(i)
                for j in range(len(Acy[0])):
                    if Acy[i,j]!=0:
                        columns_to_remove.append(j)
        Acy = np.delete(Acy,rows_to_remove,0)
        Acy = np.delete(Acy,columns_to_remove,1)
        Acx = np.delete(Acx,rows_to_remove,0)
        Acz = np.delete(Acz,rows_to_remove,0)
        y_hat = np.delete(y_hat,columns_to_remove,0)
        
        y_new = np.subtract(np.subtract(y_hat,Acx@(x_new-x_hat)),Acz)
        
        Qx = np.subtract(np.identity(len(Q)),Q@np.transpose(Arx)@np.linalg.inv(Arx@Q@np.transpose(Arx))@Arx)@Q
        Qy = Acx@ Qx @np.transpose(Acx)
        
        x_old = copy.deepcopy(x_hat)
        y_old = copy.deepcopy(y_hat)        
        
        iterations = 1
        while np.linalg.norm(np.subtract(x_new,x_old))>tol or np.linalg.norm(np.subtract(y_new,y_old))>tol:
            
            y_replace = [y[i] for i in range(len(y))]
            count = 0
            for i in range(len(y)):
                if i in columns_to_remove:
                    y_replace[i]=subs[i][1] 
                else:
                    y_replace[i]=y_new[count][0]
                    count+=1
                    
            subs = [ (y[i],y_replace[i]) for i in range(len(y))]
            subs.extend( [ (x[i],x_new[i][0]) for i in range(len(x))] )
            subs.extend( [ (key, z[key]) for key in list(z.keys())] )
            
            
            for i in range(len(EquationsDict)):
                for j in range(len(x)):
                    J_x_hat[i][j] = J_x[i][j].subs(subs)
                    
            EquationsList = list(EquationsDict.values())
            f_hat = [sym.sympify((EquationsList[i])).subs(subs) for i in range(len(EquationsList))]
           
            A = np.hstack( (np.array(J_y_hat),np.array(J_x_hat),np.array(f_hat)[:,None])  )
            A_rref, Acy,Acx,Acz,Ary,Arx,Arz = get_RREF_blocks(A,len(y),len(x))
            
            variables_to_convert = [Q, x_tilde, Acy, Acx, Acz, Ary, Arx, Arz, y_old]
            variables_converted = [np.array(var, dtype=float) for var in variables_to_convert]
            Q, x_tilde, Acy, Acx, Acz, Ary, Arx, Arz, y_old = variables_converted
            
            x_old = copy.deepcopy(x_new)
            x_new = np.subtract(x_tilde ,  Q@np.transpose(Arx)@np.linalg.inv(Arx@Q@np.transpose(Arx))@(np.add(Arx@(x_tilde-x_old),Arz)))
            
            Acy = np.delete(Acy,rows_to_remove,0)
            Acy = np.delete(Acy,columns_to_remove,1)
            Acx = np.delete(Acx,rows_to_remove,0)
            Acz = np.delete(Acz,rows_to_remove,0)
            
            y_old = copy.deepcopy(y_new)
            y_new = np.subtract(np.subtract(y_old,Acx@(x_new-x_old)),Acz)
            
            iterations +=1
            if iterations==100:
                sys.exit('Number of iterations exceeded. Alghoritm did not converge.')
         
        Qx = np.subtract(np.identity(len(Q)),Q@np.transpose(Arx)@np.linalg.inv(Arx@Q@np.transpose(Arx))@Arx)@Q
        Qy = Acx@ Qx @np.transpose(Acx)
        
        Variables, Observed_values, Observed_sd, Reconciled_values, Reconciled_sd, Notes = ([] for i in range(6))
        
        y_new_extended = [list(y_new)[i][0] for i in range(len(y_new))]
        variances = np.diag(Qy)
        variances_extended = [variances[i] for i in range(len(variances))]
        for i in columns_to_remove:
            y_new_extended.insert(i,'not computed')
            variances_extended.insert(i,'not computed')
        
        for key in MFASystem.FlowDict:
            Variables.append(key)
            if MFASystem.FlowDict[key].Values != None and MFASystem.FlowDict[key].Uncert == None: # constant
                Observed_values.append(MFASystem.FlowDict[key].Values)
                Observed_sd.append('None')
                Reconciled_values.append(z[key])
                Reconciled_sd.append('None')
                Notes.append('Constant')
            elif MFASystem.FlowDict[key].Values == None:  
                Observed_values.append('None')
                Observed_sd.append('None')
                if y.index(key) in columns_to_remove:   
                    Reconciled_values.append(InitialEstimateDict[key])
                    Reconciled_sd.append('None')
                    Notes.append('Cannot reconcile. Initial estimate only. Check for mass balance inconsistencies')
                else:   
                    Reconciled_values.append(y_new_extended[y.index(key)])
                    Reconciled_sd.append(variances_extended[y.index(key)]**0.5)
                    Notes.append('Unknown, reconciled')
            else: 
                Observed_values.append(MFASystem.FlowDict[key].Values)
                Observed_sd.append(MFASystem.FlowDict[key].Uncert * MFASystem.FlowDict[key].Values)
                Reconciled_values.append(x_new[x.index(key)][0])
                Reconciled_sd.append(Qx[x.index(key)][x.index(key)]**0.5)
                Notes.append('Observed value, reconciled')
        
        for key in MFASystem.ParameterDict:
            Variables.append(key)
            if MFASystem.ParameterDict[key].Values != None and MFASystem.ParameterDict[key].Uncert == None: 
                Observed_values.append(MFASystem.ParameterDict[key].Values)
                Observed_sd.append('None')
                Reconciled_values.append(z[key])
                Reconciled_sd.append('None')
                Notes.append('Constant')
            elif MFASystem.ParameterDict[key].Values == None: 
                Observed_values.append('None')
                Observed_sd.append('None')
                if y.index(key) in columns_to_remove:   
                    Reconciled_values.append(InitialEstimateDict[key])
                    Reconciled_sd.append('None')
                    Notes.append('Cannot reconcile. Initial estimate only. Check for mass balance inconsistencies')
                else:   
                    Reconciled_values.append(y_new_extended[y.index(key)])
                    Reconciled_sd.append(variances_extended[y.index(key)]**0.5)
                    Notes.append('Unknown, reconciled')
            else: 
                Observed_values.append(MFASystem.ParameterDict[key].Values)
                Observed_sd.append(MFASystem.ParameterDict[key].Uncert * MFASystem.ParameterDict[key].Values)
                Reconciled_values.append(x_new[x.index(key)][0])
                Reconciled_sd.append(Qx[x.index(key)][x.index(key)]**0.5)
                Notes.append('Observed value, reconciled')
        
        results = pd.DataFrame({'Variable':         Variables,
                                'Observed value':   Observed_values,
                                'Observed sd':      Observed_sd,
                                'Reconciled value': Reconciled_values,
                                'Reconciled sd':    Reconciled_sd,
                                'Notes':            Notes})
        
        reconciled_subs = [ (results['Variable'][i],results['Reconciled value'][i]) for i in range(len(results))]
        MassBalance = [float(sym.sympify((MassBalanceEquationsDict[key])).subs(reconciled_subs)) for key in MassBalanceEquationsDict.keys()]
        return results, MassBalance
    
    
def MCMCDataReconciliation(MFASystem, ParametersEquationsList, UnobservedVariablesList, FreeObservedVariablesList, DependentObservedVariablesList,  PriorPDFDict, L=100000, EquationsList=None):
    '''
    Method to perform data reconciliation of non-normal observations
    based on https://doi.org/10.1080/02664763.2017.1421916.
    
    Parameters
    ----------
        MFASystem (MFAsystem): MFAsystem object from the ODYM_CLasses library.
        ParametersEquationsList (list): List of model equations involving parameters.
        UnobservedVariablesList (list): List of unobserved variables.
        FreeObservedVariablesList (list): List of observed variables, chosen as free.
        DependentObservedVariablesList (list): List of observed variables, chosen as dependent.
        PriorPDFDict (dict): Dictionary for prior distribution for the Free Observed Variables.
        L (int): Number of simulations. The default is 200000.
        EquationsList (list): List of model equations. It used if a MFASystem is not specified. The default is None

    Returns
    -------
    w_mc (array): 2D array of free observed variables. First dimension: simulations. Second dimension: variables.
    u_mc (array): 2D array of dependent observed variables. First dimension: simulations. Second dimension: variables.
    y_mc (array): 2D array of unobserved variables. First dimension: simulations. Second dimension: variables.

    '''
    
    for variable in FreeObservedVariablesList:
        if variable not in PriorPDFDict.keys():
            sys.exit(f'{variable} is a free observed variable but a prior PDF has not been assigned.' )
            
    Variables = FreeObservedVariablesList + DependentObservedVariablesList + UnobservedVariablesList
    
    if MFASystem != None:
        
        for flow in MFASystem.FlowDict:
            try:
                if len(MFASystem.FlowDict[flow].Indices) > 1:
                    sys.exit(f'Flow {flow} has more than one dimension. To perform data reconciliation of multi-layers systems, set MFASystem and ParametersEquationsList to None and pass the complete set of equations, including mass-balance.' )
            except:
                pass
             
        MassBalanceEquationsDict, EquationsDict = {},{}
        
        for key in MFASystem.ProcessList:
            MassBalanceEquationsDict[str(key.ID)] = ''
        
        for key in MFASystem.FlowDict:
            if MFASystem.FlowDict[key].P_Start == None:
                MassBalanceEquationsDict[str(MFASystem.FlowDict[key].P_End)] += '-'+str(key)
            else:
                if MFASystem.FlowDict[key].P_Start != 0:
                    MassBalanceEquationsDict[str(MFASystem.FlowDict[key].P_Start)] += '-'+str(key)
                if MFASystem.FlowDict[key].P_End != 0:
                    MassBalanceEquationsDict[str(MFASystem.FlowDict[key].P_End)]   += '+'+str(key)
        
        for key in MassBalanceEquationsDict.keys():
            EquationsDict[key] = MassBalanceEquationsDict[key]
        
        for equation in ParametersEquationsList:
            EquationsDict[list(MFASystem.ParameterDict.keys())[ParametersEquationsList.index(equation)]] = equation.split('=')[0]
        EquationsList = list(EquationsDict.values())
    
    else:
        EquationsList = [equation.split('=')[0] for equation in EquationsList]
    
    
    J = [[None for i in range(len(Variables))] for j in range(len(EquationsList))]
    for i in range(len(EquationsList)):
        for j in range(len(Variables)): 
            J[i][j] = sym.diff(EquationsList[i],Variables[j])
    
    n = len(FreeObservedVariablesList) + len(DependentObservedVariablesList) + len(UnobservedVariablesList)
    if n-len(EquationsList) != len(FreeObservedVariablesList):
        sys.exit(f"Inconsistent system and variables. With {n} variables and {len(EquationsList)} constraints (model equations: {len(MassBalanceEquationsDict)} mass-balance equations and {len(ParametersEquationsList)} parameters equations), {n-len(EquationsList)} free observed variables are expected.")
    
    try: 
        solution_dict = solve(EquationsList, tuple([sym.Symbol(key) for key in (UnobservedVariablesList+DependentObservedVariablesList) ]), dict=True)[0]
    except:
        sys.exit('Not a good choice for the free observed variables. Cannot solve the system.')
    
    solution_dict = solve(EquationsList, tuple([sym.Symbol(key) for key in (UnobservedVariablesList+DependentObservedVariablesList) ]), dict=True)[0]
        
    h, k = {}, {}
    for key in DependentObservedVariablesList:
        h[key] = solution_dict[sym.Symbol(key)]
    for key in UnobservedVariablesList:
        k[key] = solution_dict[sym.Symbol(key)]
    
    # IS-NL2: Generate the starting value w1 by drawing independent random numbers from the prior marginal distributions of the free variables
    w_i = [ pdf(PriorPDFDict[key]).sample() for key in FreeObservedVariablesList  ]
    subs_wi = [ ( FreeObservedVariablesList[i], w_i[i] ) for i in range(len(FreeObservedVariablesList))  ]
    h_wi = [ h[key].subs(subs_wi) for key in list(h.keys())]
    
    itera = 1
    while np.prod(h_wi)<=0:
        w_i = [ pdf(PriorPDFDict[key]).sample() for key in FreeObservedVariablesList  ]
        subs_wi = [ ( FreeObservedVariablesList[i], w_i[i] ) for i in range(len(FreeObservedVariablesList))  ]
        h_wi = [ h[key].subs(subs_wi) for key in list(h.keys())]
        itera +=1
        if itera==100:
            sys.exit('Number of iterations exceeded. Alghoritm did not converge.')
     
    # IS-NL3 Generate a proposal value w˙ by drawing independent random numbers from the prior marginal distributions of the free variables
    w_dot = [ pdf(PriorPDFDict[key]).sample() for key in FreeObservedVariablesList  ]
    subs_w_dot = [ ( FreeObservedVariablesList[i], w_dot[i] ) for i in range(len(FreeObservedVariablesList))  ]
    h_w_dot = [ h[key].subs(subs_w_dot) for key in list(h.keys())]
    
    # IS-NL4  Compute the acceptance probability α
    H = [[None for i in range(len(FreeObservedVariablesList))] for j in range(len(h))]
    for i in range(len(h)):
        for j in range(len(FreeObservedVariablesList)): 
            H[i][j] = sym.diff(list(h.values())[i],FreeObservedVariablesList[j])
            
    Hnp = np.array(H)
    arg = np.add(np.identity((Hnp.transpose()@H).shape[0]), Hnp.transpose()@H) 
    arg_wi = np.array(sym.sympify(arg).subs(subs_wi), dtype=float)
    arg_w_dot = np.array(sym.sympify(arg).subs(subs_w_dot), dtype=float)
    V_wi = np.linalg.det(arg_wi)**0.5
    V_w_dot = np.linalg.det(arg_w_dot)**0.5
    
    alpha = min(1, (np.prod(h_w_dot)*V_w_dot)/(np.prod(h_wi)*V_wi))
    
    # IS-NL5 Draw a uniform random number ξ ∈ [0, 1].
    xi = sts.uniform.rvs(0,1)
    # IS-NL6 If ξ ≤ α, accept the proposal and set wi+1 = w˙ , otherwise set wi+1 = wi.
    if xi <= alpha:
        w_i = w_dot        
    
    w_mc, u_mc, y_mc = [], [], []
    w_mc.append(w_i)
    subs_wi = [ ( FreeObservedVariablesList[i], w_i[i] ) for i in range(len(FreeObservedVariablesList))  ]
    h_wi = [ h[key].subs(subs_wi) for key in list(h.keys())]
    u_mc.append(h_wi)
    k_wi = [ k[key].subs(subs_wi) for key in list(k.keys())]
    y_mc.append(k_wi)
    
    # IS-NL7 iterations
    for iterations in tqdm(range(L)):
        w_dot = [ pdf(PriorPDFDict[key]).sample() for key in FreeObservedVariablesList  ]
        subs_w_dot = [ ( FreeObservedVariablesList[i], w_dot[i] ) for i in range(len(FreeObservedVariablesList))  ]
        h_w_dot = [ h[key].subs(subs_w_dot) for key in list(h.keys())]
        
        arg_wi = np.array(sym.sympify(arg).subs(subs_wi), dtype=float)            
        arg_w_dot = np.array(sym.sympify(arg).subs(subs_w_dot), dtype=float)
        
        V_wi = np.linalg.det(arg_wi)**0.5
        V_w_dot = np.linalg.det(arg_w_dot)**0.5
        
        alpha = min(1, (np.prod(h_w_dot)*V_w_dot)/(np.prod(h_wi)*V_wi))
        xi = sts.uniform.rvs(0,1)
        if xi <= alpha:
            w_i = w_dot        
        
        w_mc.append(w_i)
        subs_wi = [ ( FreeObservedVariablesList[i], w_i[i] ) for i in range(len(FreeObservedVariablesList))  ]
        h_wi = [ h[key].subs(subs_wi) for key in list(h.keys())]
        u_mc.append(h_wi)
        k_wi = [ k[key].subs(subs_wi) for key in list(k.keys())]
        y_mc.append(k_wi)
        
    w_mc = np.array(w_mc, float)  
    u_mc = np.array(u_mc, float)
    y_mc = np.array(y_mc, float)
    
    return w_mc, u_mc, y_mc

    
def get_RREF_blocks(M,ny,nx):
    '''
    Function that transform a matrix into Row Echelon Form (RREF) for Data Reconciliation purpose. 
    See https://doi.org/10.1016/j.serj.2016.06.002 for notation.
    
    Parameters
    ----------
        M (array): 2D array of the matrix to be transformed.
        ny (int): Number of columns of submatrix Mcy.
        nx (int): Number of columns of submatrix Mcx
    
    Returns
    -------
        M_rref (array): 2D array of RREF matrix. 
        Mcy,Mcx,Mcz,Mry,Mrx,Mrz (arrays): 2D arrays of the sub-matrixes of R_rref used in DataREconciliation method.
            
    '''
    M_conv = sym.Matrix(M)
    M_rref = M_conv.rref()[0]
    M_rref = np.array(M_rref).astype(np.float64)
    nrows = len(M)
    ncols = len(M[0])
    i=0
    for i in range(nrows):
        if [ M_rref[i,j] for j in range(0,ny) ] == [ 0 for j in range(0,ny) ]:
            i_split = i
            break
    Mcy = M_rref[0:i_split,0:ny]
    Mcx = M_rref[0:i_split,ny:ny+nx]
    Mcz = M_rref[0:i_split,ny+nx:ncols]
    Mry = M_rref[i_split:nrows,0:ny]
    Mrx = M_rref[i_split:nrows,ny:ny+nx]
    Mrz = M_rref[i_split:nrows,ny+nx:ncols]
    return M_rref, Mcy,Mcx,Mcz,Mry,Mrx,Mrz

    
def check_system_solution(MFASystem,ParametersEquationsList,UnobservedVariablesList,DependentObservedVariablesList ):
    '''   
    Function that checks whether the system can be solved with the selected choice of variables.
    
    Parameters
    ----------
        MFASystem (MFAsystem): MFAsystem object from ODYM_CLasses library.
        ParametersEquationsList (list): List of model equations of the system involving parameters.
        UnobservedVariablesList (list): List of unobserved system variables.
        DependentObservedVariablesList (list): List of observed system variables, chosen as dependent.

    Returns
    -------
        A string that states whether the system can be solved with the selected choice of variables.

    '''
    EquationsDict = {}
    for key in MFASystem.ProcessList:
        EquationsDict[str(key.ID)] = ''
    for key in MFASystem.FlowDict:
        if MFASystem.FlowDict[key].P_Start == None:
            EquationsDict[str(MFASystem.FlowDict[key].P_End)] += '-'+str(key)
        else:
            if MFASystem.FlowDict[key].P_Start != 0:
                EquationsDict[str(MFASystem.FlowDict[key].P_Start)] += '-'+str(key)
            if MFASystem.FlowDict[key].P_End != 0:
                EquationsDict[str(MFASystem.FlowDict[key].P_End)]   += '+'+str(key)
    for equation in ParametersEquationsList:
        EquationsDict[list(MFASystem.ParameterDict.keys())[ParametersEquationsList.index(equation)]] = equation.split('=')[0]
    EquationsList = list(EquationsDict.values())
    
    try: 
        solve(EquationsList, tuple([sym.Symbol(key) for key in (UnobservedVariablesList+DependentObservedVariablesList) ]), dict=True)[0]
        return 'The system can be solved with this set of variables!'
    except:
        return 'Not a good choice for the free observed variables. Cannot solve the system.'
    
    
def plot_density(data,bins,title='',density_only=False,ax=None,prior=False):
    '''
    Method that plots the distribution of samples.
    
    Parameters
    ----------
        data (array): 1D array with samples to plot.
        bins (int): Number of bins for the histogram plot. 
        title (str): Title of the plot. The default is the empty string ''.
        density_only (boolean): If True, histogram bars are discarded and only the smoothed values are plotted. The default is False.
        ax (axes): selected ax for subplots methods. The default is None. 
        prior (boolean): If True, the legend for the prior distribution is imported. Note: it does NOT plot the prior, which had to be previously plotted outside the function.
        
    '''
    if ax == None:
        ax = plt.gca()
    if density_only == False:
        sns.histplot(data=data, bins=bins, stat='density', kde=False, edgecolor='white', linewidth=0.5,ax=ax)
        sns.kdeplot( data=data,  color='blue',ax=ax)
        ax.legend(['Posterior (KDE)'], loc = 'upper right')
    elif prior == False:
        sns.kdeplot( data=data, color='blue',ax=ax)
        ax.legend(['Posterior (KDE)'], loc = 'upper right')
    else:
        h1, l1 = ax.get_legend_handles_labels()
        sns.kdeplot( data=data, color='blue',ax=ax)
        h2, l2 = ax.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2)
        ax.legend(['Prior','Posterior (KDE)'], loc = 'upper right')
    ax.set_title(title)
    
    
class pdf:
    '''
    Class for Probability Density Function.
    '''
    def __init__(self, pdf_dict):
        self.name = pdf_dict['name']
        self.parameters_dict = pdf_dict['parameters']
        
    def sample(self):
        '''' Returns a random value sampled from the probability density'''
        if(self.name == "normal"):
            self.mean = self.parameters_dict['mean']
            self.sd   = self.parameters_dict['sd']
            self.random_value  = sts.norm.rvs(loc = self.mean, scale = self.sd) 
            return self.random_value
        elif(self.name == "uniform"): 
            self.min = self.parameters_dict['min']
            self.max = self.parameters_dict['max']
            self.random_value  = sts.uniform.rvs(loc = self.min, scale = self.max-self.min) 
            return self.random_value
        elif(self.name == "triang"):
            self.min = self.parameters_dict['min']
            self.max = self.parameters_dict['max']
            self.c   = self.parameters_dict['c']
            self.random_value  = sts.triang.rvs(c = (self.c-self.min)/(self.max-self.min), loc = self.min, scale = self.max-self.min) 
            return self.random_value
        elif(self.name == "truncnorm"):            
            self.min  = self.parameters_dict['min']
            self.max  = self.parameters_dict['max']            
            self.mean = self.parameters_dict['mean']
            self.sd = self.parameters_dict['sd']
            self.random_value  = sts.truncnorm.rvs(a = (self.min-self.mean)/self.sd, b = (self.max-self.mean)/self.sd, loc = self.mean, scale = self.sd) 
            return self.random_value
        else: # 
            try:
                self.random_value = getattr(sts,self.name).rvs(**self.parameters_dict)
                return self.random_value
            except:
                sys.exit(f'{self.name} distribution not properly initiated. Check https://docs.scipy.org/doc/scipy/reference/stats.html')
                
# The end
