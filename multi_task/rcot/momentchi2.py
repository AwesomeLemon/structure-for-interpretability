"""
momentchi2 package which rcot uses:
https://github.com/cran/momentchi2/tree/master/R
"""
import numbers
from scipy.special import factorial, binom
from scipy import stats
import numpy as np

def sw(coeff, x):
    """
    https://github.com/cran/momentchi2/blob/master/R/sw.R
    """
    if isinstance(coeff, numbers.Number): 
        assert coeff > 0
    else:
        coeff = np.array(coeff)
        #assert np.sum(coeff>=0) == len(coeff)
    if isinstance(x, numbers.Number):
        assert x > 0
    else:
        x = np.array(x)
        assert np.sum(x>0) == len(x)

    w_val = np.sum(coeff)
    u_val = np.sum(np.square(coeff)) / (w_val**2)
	
	#now the G k and theta:
    gamma_k = 0.5 / u_val
    gamma_theta = 2 * u_val*w_val
	
	#the actual x value
    # pgamma in R is equivalent to 1- survival function in scipy
    # cast to make sure:
    x = np.real(x)
    p_sw = 1-stats.gamma.sf(x, gamma_k, scale=gamma_theta)	
    return p_sw


def hbe(coeff, x):
    """
    https://github.com/cran/momentchi2/blob/master/R/hbe.R
    """
    if isinstance(coeff, numbers.Number): 
        assert coeff > 0
    else:
        coeff = np.array(coeff)
        #assert np.sum(coeff>=0) == len(coeff)
    if isinstance(x, numbers.Number):
        assert x > 0
    else:
        x = np.array(x)
        assert np.sum(x>0) == len(x)

    kappa = np.array([np.sum(coeff), 2*np.sum(np.square(coeff)), 8*np.sum(np.power(coeff,3))])
    K_1 = kappa[0]
    K_2 = kappa[1]
    K_3 = kappa[2]
    nu = 8 * (K_2**3) / (K_3**2)
    
    #gamma parameters for chi-square
    gamma_k = nu/2
    gamma_theta = 2

    #need to transform the actual x value to x_chisqnu ~ chi^2(nu)
    #This transformation is used to match the first three moments
    #First x is normalised and then scaled to be x_chisqnu
    x_chisqnu_vec = np.sqrt(2 * nu / K_2) * (x - K_1) + nu
    
    #now this is a chi_sq(nu) variable
    # cast to make sure
    p_chisqnu_vec = 1-stats.gamma.sf(np.real(x_chisqnu_vec), np.real(gamma_k), scale=gamma_theta)	
    return p_chisqnu_vec



"""
helper functions for lpb4:
"""

def get_cumulant_vec_vectorised(coeffvec, p):
    index = np.array(range(2*p))
    #now cumulants are multiplied by sum_of_powers of coeffvec
    #moment_vec <- cumulants * vapply(X=v, FUN=sum_of_powers, FUN.VALUE=rep(0,1), v=coeffvec)	
    cumulant_vec = 2**index * factorial(index) * np.array([np.sum(coeffvec**(i+1)) for i in index])
    return cumulant_vec

def update_moment_from_lower_moments_and_cumulants(n, moment_vec, cumulant_vec):
    m = np.array(range(n))
    sum_of_additional_terms = np.sum(binom(n, m) * cumulant_vec[m] * moment_vec[n-m-1])
    return sum_of_additional_terms

def get_moments_from_cumulants(cumulant_vec):
    #start off by assigning it to cumulant_vec, since moment[n] = cumulant[n] + {other stuff}
    moment_vec = np.copy(cumulant_vec)
    #check if more than 1 moment required
    if len(moment_vec)>1:
        #can't get rid of this for loop, since updates depend on previous moments
        for n in range(1, len(moment_vec)):
            #can vectorise this part, I think
            moment_vec[n] += update_moment_from_lower_moments_and_cumulants(n, moment_vec, cumulant_vec) 					
    return moment_vec

#hides the computation of the cumulants, by just talking about moments
def get_weighted_sum_of_chi_squared_moments(coeffvec, p):
    cumulant_vec = get_cumulant_vec_vectorised(coeffvec, p)
    moment_vec = get_moments_from_cumulants(cumulant_vec)
    return moment_vec


#no need to use bisection method - can get lambdatilde_1 directly 
def get_lambdatilde_1(m1, m2):
	return m2/(m1**2) - 1 


#get_partial_products gets prod[1:index] 
def get_partial_products(index, vec):
    return np.prod(vec[:index]) 


#this function in deltaNmat_applied computes the index from i and j, and then returns the appropriate product
#of vec1 and vec2 
#(in deltaNmat_applied, these vectors are the moment vector and the vector of the products of the (1+N*lambda)^(-1) terms)
def get_index_element(i, j, vec1, vec2):
	index = i + j
	return vec1[index] * vec2[index]


#compute the delta_N matrix - vectorised using lapply and mapply
def deltaNmat_applied(x, m_vec, N):
    Nplus1 = N+1
    #want moments 0, 1, ..., 2N
    m_vec = np.array([1] + list(m_vec[0:(2*N)]))

    #these will be the coefficients for the x in (1+c_1*x)*(1+c_2*x)*...
    #want coefficients 0, 0, 1, 2, .., 2N-1 - so 2N+1 in total 
    coeff_vec = np.array([0] + list(range(int(2*N))))*x + 1
    #not necessary to initialise, could use length(m_vec) below, but we do it anyway for readability
    prod_x_terms_vec = np.zeros(2*N+1)
    #this computes the terms involving lambda in a vectorised way
    prod_x_terms_vec = 1/np.array([get_partial_products(i+1, coeff_vec) for i in range(2*N+1)])
	
    #going to use mapply over matrix indices i, j
    i_vec = np.array(range(Nplus1))
    j_vec = np.array(range(Nplus1))


    #not necessary to initialise
    #delta_mat <- matrix(0, Nplus1, Nplus1)
    delta_mat = np.zeros((Nplus1, Nplus1))
    for i in i_vec:
        delta_mat[i,:] = get_index_element(i, j_vec, m_vec, prod_x_terms_vec)

    return delta_mat


#Simply uses above matrix generation function
def det_deltamat_n(x, m_vec, N):
	return np.linalg.det(deltaNmat_applied(x, m_vec, N)) 


from scipy.optimize import brentq as root

#Step 3: get lambdatilde_p
#uses det_delta_mat_n and uniroot

#get lambdatilde_p by using bisection method repeatedly. 
#Need lambdatilde_1 to start
#Need to use R function uniroot
def get_lambdatilde_p(lambdatilde_1, p, moment_vec):
	lambdatilde_vec = np.zeros(p)
	lambdatilde_vec[0] = lambdatilde_1 
	bisect_tol = 1e-9
	
	#check that p>1
	if p > 1:
		for i in range(1,p):
			r = root(lambda x: det_deltamat_n(x, m_vec=moment_vec, N=i+1), a=0, b=lambdatilde_vec[i-1], xtol=bisect_tol, rtol=bisect_tol)
			lambdatilde_vec[i] = r	
		#end of for		
	#now distinguish lambdatilde_p
	lambdatilde_p = lambdatilde_vec[-1]
	return lambdatilde_p


#Step 5.2: Compute polynomial coefficients for mu polynomial

#generate a base vector of all zeros except for 1 in ith position
def get_base_vector(n, i):
	base_vec = np.zeros(n)
	base_vec[i] = 1
	return base_vec



#get the ith coefficient by computing determinant of appropriate matrix
def get_ith_coeff_of_Stilde_poly(i, mat):
    n = mat.shape[0]
    base_vec = get_base_vector(n, i)
    mat[:, n-1] = base_vec
    return np.linalg.det(mat)


#We could use the linear algebra trick described in the Lindsay paper, but want to avoid
#dealing with small eigenvalues. Instead, we simply compute p+1 determinants.
#
#This method replaces last column with the base vectors (0, ..., 0 , 1, 0, ... 0) 
#to compute the coefficients, and so does not need to compute 
#any eigen decomposition, just (p+1) determinants
def get_Stilde_polynomial_coefficients(M_p):
    #number of rows, number of coefficients ( ==(p+1) )
    n = M_p.shape[0]
    mu_poly_coeff_vec = np.array([get_ith_coeff_of_Stilde_poly(i, M_p) for i in range(n)])
    return mu_poly_coeff_vec


#simply takes the last column, and removes last element of last column
def get_VDM_b_vec(mat):
	b_vec = mat[:, 0]
	b_vec = b_vec[:-1]
	return b_vec

#generates the van der monde matrix from a vector
def generate_van_der_monde(vec):
	p = vec.shape[0]
	vdm = np.zeros((p,p))
	for i in range(p):
		vdm[i, :] = vec**i
	return vdm

#Step 6:Generate van der monde (VDM) matrix and solve the system VDM * pi_vec = b

#generates the VDM matrix and solves the linear system. 
#uses R's built in solve function - there may be a better VDM routine (as cited in Lindsay)
def generate_and_solve_VDM_system(M_p, mu_roots):
	#easiest way to get rhs vector is to just take first column of M_p
	b_vec = get_VDM_b_vec(M_p)
	#print(b_vec)
	#generate Van der Monde matrix; just powers of mu_roots
	VDM = generate_van_der_monde(mu_roots)
	#print(VDM)
	
	#cat("pi_vec:\n")
	#use R's solve function to solve the linear system
	#there may be better routines for this, but such an implementation is deferred until later
	#NB: If p is too large (p>10), this can yield an error (claims the matrix is singluar).
	#A tailor-made VDM solver should fix this.
	pi_vec = np.linalg.solve(VDM, b_vec)
	return pi_vec


#Step 7: Here we use mu_vec, pi_vec and lambdatilde_p to compute the composite pgamma values 
#		 and combine them into the ifnal pvalue


#computes pgamma of the appropriate gamma function
def compute_composite_pgamma(index, qval, shape_val, scale_vec):
	return 1-stats.gamma.sf(qval, shape_val, scale=scale_vec[index]) 

#get_mixed_p_val - weight sum of pgammas
#now compute for a vector of quantiles - assume the vector of quantiles is very long,
#while p < 10 (so vectorise over length of quantiles)
def get_mixed_p_val_vec(quantile_vec, mu_vec, pi_vec, lambdatilde_p):
    #First compute the composite pvalues
    p = mu_vec.shape[0]
    
    if isinstance(quantile_vec, np.ndarray):
        l = quantile_vec.shape[0]
    else:
        l = 1
	
	#For pgamma, we need to specify the shape and scale parameters
	#shape alpha = 1/lambda
    alpha = 1/lambdatilde_p
	#NB: scale beta = mu/alpha, as per formulation in Lindsay paper
    beta_vec = mu_vec/alpha
	
	#we could probably vectorise this, but this is simpler
	#we use the pgamma to compute a vector of pvalues from the vector of quantiles, for a given distribution
	#we then scale this by the appropriate pi_vec value, and add this vector to a 0 vector, and repeat
	#finally, each component of the vector is a pi_vec-scaled sum of pvalues
    partial_pval_vec = np.zeros(l)
    for i in range(p):
        partial_pval_vec += pi_vec[i] * (1-stats.gamma.sf(np.real(quantile_vec), np.real(alpha), scale=np.real(beta_vec[i])))		
	
    if l == 1:
        partial_pval_vec = partial_pval_vec[0]
    return partial_pval_vec


"""
end helper functions
"""

def lpb4(coeff, x):
    """
    https://github.com/cran/momentchi2/blob/master/R/lpb4.R
    """
    if isinstance(coeff, numbers.Number): 
        assert coeff > 0
    else:
        coeff = np.array(coeff)
        #assert np.sum(coeff>=0) == len(coeff)
    if isinstance(x, numbers.Number):
        assert x > 0
    else:
        x = np.array(x)
        assert np.sum(x>0) == len(x)

    if len(coeff) < 4:
        print("Less than four coefficients - LPB4 method may return NaN: running hbe instead.")
        return hbe(coeff, x)

    #----------------------------------------------------------------#
    #step 0: decide on parameters for distribution and support points p
    #specified to be 4 for this version of the function
    p = 4
    
    #----------------------------------------------------------------#
    #step 1: Determine/compute the moments m_1(H), ... m_2p(H)
    
    #compute the first 2p moments for Q = sum coeff chi-squared

    moment_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)
    
    #----------------------------------------------------------------#
    #Step 2.1: generate matrices delta_i(x)
    #functions created:
    #deltaNmat_applied
    #and
    #det_deltamat_n
    
    #Step 2.2: get lambdatilde_1 - this method is exact (no bisection), solves determinant equation
    lambdatilde_1 = get_lambdatilde_1(moment_vec[0], moment_vec[1])
    
    #----------------------------------------------------------------#
    #Step 3:	Use bisection method (R uniroot) to find lambdatilde_2 
    #and others up to lambdatilde_p, for tol=bisect_tol
    #all we need is the final lambdatilde_p, not the intermediate values lambdatilde_2, lambdatilde_3, etc
    lambdatilde_p = get_lambdatilde_p(lambdatilde_1, p, moment_vec)

    #----------------------------------------------------------------#
    #Step 4:
    #Calculate delta_star_lambda_p
    #can already do this using methods in Step 2.1 
    #----------------------------------------------------------------#
    
    #----------------------------------------------------------------#
    #Step 5:
    
    #Step 5.1: use the deltastar_i(lambdatilde_p) from Step 4 to generate
    #			M_p, which will be used to create matrix Stilde(lambdatilde_p, t)
    M_p = deltaNmat_applied(lambdatilde_p, moment_vec, p)
    
    #Step 5.2:	Compute polynomial coefficients of the modified M_p matrix (as in paper).
    mu_poly_coeff_vec = get_Stilde_polynomial_coefficients(M_p)

    #step 5.3	use Re(polyroot(coeff_vec)) to obtain roots of polynomial
    #			denoted mu_vec = (mu_1, ..., mu_p)	
    mu_roots = np.real(np.polynomial.polynomial.polyroots(mu_poly_coeff_vec))
    
    #----------------------------------------------------------------#

    #Step 6:	Generate Vandermonde matrix using mu_vec
    #			and vector using deltastar_i's, to solve for
    #			pi_vec = (pi_1, ..., pi_p)
    pi_vec = generate_and_solve_VDM_system(M_p, mu_roots)
    
    #----------------------------------------------------------------#
    
    #Step 7: 	Compute the linear combination (using pi_vec)
    #			of the i gamma cdfs using parameters lambdatilde_p and mu_i 
    #			(but need to create scale/shape parameters carefully)
    #	
    #			This is the final answer
    
    mixed_p_val_vec = get_mixed_p_val_vec(x, mu_roots, pi_vec, lambdatilde_p)
    
    #We have our final answer, and so return it
    return mixed_p_val_vec
"""
end momentchi2
"""