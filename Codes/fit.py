from cfpack import print, stop
import numpy as np

def myfit(func, xdat, ydat, xerr=None, yerr=None, perr_method='statistical', n_random_draws=1000, dat_frac_for_systematic_perr=0.6,
        weights=None, scale_covar=False, params=None, verbose=1, *args, **kwargs):

    from lmfit import Model as lmfit_model
    model = lmfit_model(func) # get lmfit model object
    # set up parameters
    lmfit_params = model.make_params() # make lmfit default params

    if params is not None: # user previded parameter bounds and initial values
        for pname in params:
            if pname not in model.param_names: print("parameter key error: '"+pname+"'", error=True)
            plist = params[pname]
            #value = (plist[1]-plist[0]) / 2 # initial parameter value
            '''the next two lines are Lakshmi's addition; just for your reference; please feel free to delete/alter this line for public addition'''
            if None in plist: model.set_param_hint(pname, value=plist[1], vary=False)
            else: model.set_param_hint(pname, min=plist[0], value=plist[1], max=plist[2]) # set bounds for this parameter from [min, val, max]

    else: # try and find some reasonable initial guesses for the fit parameters
        if verbose: print("trying to find initial parameter guesses...")
        # genetic algorithm to come up with initial fit parameter values
        def generate_initial_params(n_params):
            from scipy.optimize import differential_evolution
            # function for genetic algorithm to minimize (sum of squared error)
            def sum_of_squared_error(parameterTuple):
                from warnings import filterwarnings
                filterwarnings("ignore") # do not print warnings by genetic algorithm
                val = func(xdat, *parameterTuple)
                return np.sum((ydat - val)**2)
            # min and max used for bounds
            maxX = np.max(xdat)
            maxY = np.max(ydat)
            maxXY = np.max([maxX, maxY])
            parameterBounds = [[-maxXY, maxXY]]*n_params
            result = differential_evolution(sum_of_squared_error, parameterBounds, seed=140281)
            return result.x
        # get initial parameter value guesses and put them in
        initial_params = generate_initial_params(len(model.param_names))
        for ip, pname in enumerate(model.param_names):
            model.set_param_hint(pname, value=initial_params[ip])

    # re-create fit parameter settings after parameter bounds or initial guesses have been determined
    lmfit_params = model.make_params() # update lmfit params
    # get initial parameter info
    if verbose > 1:
        for pname in lmfit_params:
            print("parameters (start): ", lmfit_params[pname])
    # prepare for return class
    ret_popt = [] # parameter best value
    ret_pstd = [] # parameter standard deviation
    ret_perr = [] # parameter error range (lower and upper)
    ret_psamples = [] # parameter sampling list (only if xerr or yerr is used)
    ret_lmfit_result = None # lmfit result
    # dealing with weights or errors (1-sigma data errors in x and/or y)
    if weights is not None and ((xerr is not None) or (yerr is not None)):
        print("cannot use weights when either xerr or yerr is present", error=True)

    if xerr is not None or yerr is not None or perr_method == "systematic":
        popts = [] # list of parameter samples
        if perr_method == "statistical":
            print("Performing statistical error estimate with "+str(n_random_draws)+" sampling fits, based on data errors provided...", highlight=True)
            # draw from Gaussian random distribution
            if xerr is not None:
                xtry = np.array([generate_random_gaussian_numbers(n=n_random_draws, mu=xdat[i], sigma=xerr[i], seed=None) for i in range(len(xdat))])
            if yerr is not None:
                ytry = np.array([generate_random_gaussian_numbers(n=n_random_draws, mu=ydat[i], sigma=yerr[i], seed=None) for i in range(len(ydat))])
            # for each random sample, fit and record the best-fit parameter(s) in popts
            for i in range(n_random_draws):
                if xerr is not None:
                    x = xtry[:,i]
                else:
                    x = xdat
                if yerr is not None:
                    y = ytry[:,i]
                else:
                    y = ydat
                independent_vars_dict = {model.independent_vars[0]:x} # set independent variable
                fit_result = model.fit(data=y, params=lmfit_params, weights=None, *args, **kwargs, **independent_vars_dict)
                popt = []
                for pname in fit_result.params:
                    popt.append(fit_result.params[pname].value)
                popts.append(popt)

        if perr_method == "systematic":
            print("Performing systematic error estimate with "+str(n_random_draws)+" sampling fits, based on random subsets of "+str(dat_frac_for_systematic_perr*100)+"% of the original data...", highlight=True)
            from random import seed, randrange
            n_dat_frac = max([len(lmfit_params)+1, int(np.ceil(dat_frac_for_systematic_perr*len(xdat)))]) # take only a fraction of the original data size (minimally, the number of parameters + 1)
            n_dat_toss = len(xdat) - n_dat_frac # number of data elements to drop
            for i in range(n_random_draws):
                x = np.copy(xdat) # copy original x data
                y = np.copy(ydat) # copy original y data
                for i in range(n_dat_toss): # now randomly remove indices
                    seed(None) # set the random seed; if None, random uses the system time
                    ind = randrange(len(x))
                    x = np.delete(x, ind)
                    y = np.delete(y, ind)
                    stop()
                independent_vars_dict = {model.independent_vars[0]:x} # set independent variable
                fit_result = model.fit(data=y, params=lmfit_params, weights=None, *args, **kwargs, **independent_vars_dict)
                popt = []
                for pname in fit_result.params:
                    popt.append(fit_result.params[pname].value)
                popts.append(popt)

        # prepare return values (median, standard deviation, 16th to 84th percentile, and complete list of popts)
        if len(popts) > 0:
            popts = np.array(popts)
            for ip, pname in enumerate(fit_result.params):
                median = np.percentile(popts[:,ip], 50)
                percentile_16 = np.percentile(popts[:,ip], 16)
                percentile_84 = np.percentile(popts[:,ip], 84)
                ret_popt.append(median)
                ret_pstd.append(np.std(popts[:,ip]))
                ret_perr.append(np.array([-(median-percentile_16), (percentile_84-median)]))
                ret_psamples.append(popts)

    else: # do a normal weighted or unweighted fit
        weights_info_str = "without"
        if weights is not None: weights_info_str = "with"
        print("Performing normal fit "+weights_info_str+" weights...", highlight=True)
        if weights is not None and scale_covar:
            print("Assuming good fit for reporting parameter errors. "+
                    "Consider setting 'scale_covar=False' if you believe the fit is not of good quality.", warn=True)
        # do the fit
        independent_vars_dict = {model.independent_vars[0]:xdat} # set independent variable
        fit_result = model.fit(data=ydat, params=lmfit_params, weights=weights, scale_covar=scale_covar, *args, **kwargs, **independent_vars_dict)
        ret_lmfit_result = fit_result # for return class below
        # prepare return values
        for ip, pname in enumerate(fit_result.params):
            ret_popt.append(fit_result.params[pname].value)
            ret_pstd.append(fit_result.params[pname].stderr)
            if ret_pstd[-1] is not None:
                ret_perr.append(np.array([-fit_result.params[pname].stderr, fit_result.params[pname].stderr]))
            else:
                ret_perr.append(None)
            ret_psamples.append(None)

    class ret: # class object to be returned
        lmfit_result = ret_lmfit_result # lmfit object
        pnames = model.param_names # parameter names (list)
        popt = ret_popt # parameter best-fit values (list)
        pstd = ret_pstd # parameter standard deviation values (list)
        perr = ret_perr # parameter errors; upper and lower value (list*2)
        psamples = ret_psamples # parameter sampling list
    if verbose:
        for ip, pname in enumerate(ret.pnames):
            print("fit parameters: ", pname+" = ", ret.popt[ip], ret.perr[ip], highlight=True)
    return ret