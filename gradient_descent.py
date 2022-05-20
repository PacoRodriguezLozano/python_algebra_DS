def gradient_descent(X, y, theta0 = 0, theta1 = 0, eps = 0.01, max_iter = 10000, alpha = 0.01):
    '''
    This function implements gradient descent to perform a linear fit.
    Returns a dictionary of all thetas, costs and total iters.
    '''
    def cost_function(X,y):
        return lambda theta0, theta1: np.sum(((theta0 + theta1 * X)-y)**2)/len(X)

    def deriv_th0(X, y):
        return lambda theta0, theta1: 2/len(X) * np.sum(theta0 + theta1 * X -y)

    def deriv_th1(X, y):
        return lambda theta0, theta1: 2/len(X) * np.sum((theta0 + theta1 * X -y) * X)
    
    J = cost_function(X,y)
    Dtheta0 = deriv_th0(X,y)
    Dtheta1 = deriv_th1(X,y)
    cost = J(theta0, theta1)
    
    
    summary_theta = [[theta0, theta1]]
    summary_cost = [cost]
    
    for i in range(max_iter):
        th0_ = theta0 - alpha * Dtheta0(theta0, theta1)
        th1_ = theta1 - alpha * Dtheta1(theta0, theta1)
        
        cost_ = J(th0_,th1_)
        
        deltaCost = abs(cost - cost_)
        
        theta0 = th0_
        theta1 = th1_
        cost = cost_
        
        summary_theta.append([theta0, theta1])
        summary_cost.append(cost)
        
        if deltaCost < eps:
            break
            
    iters = len(summary_cost)-1
    results = {'thetas': summary_theta,
               'costs' : summary_cost,
               'iters' : iters
              }
    return results
                
