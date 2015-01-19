from gibbs import *

X = np.random.randn(1000).reshape((500,2))
b = [3,10]
mean_y = np.dot(X,b)
sig_squared = 1
n = 500
#construct c_p
p =.5
C_p = np.empty((n,n))
#fill it with the powers of p
for i in range(0,n):
    for j in range(0,n):
        C_p[i,j] = p**(max(i-j,j-i))
        
y = stats.multivariate_normal(mean = mean_y, cov = sig_squared*C_p).rvs()


#run the sampler
samps=Gibbs().sample(y,X, 10000)

#save the hists
import matplotlib.pyplot as plt

def return_hist(parameter_vec):
    plt.clf()
    hist = np.histogram(parameter_vec, bins=20)
    hist, bins = np.histogram(parameter_vec, bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    return plt

#save histograms
return_hist([samp[0][0] for samp in samps]).savefig("beta_1")
return_hist([samp[0][1] for samp in samps]).savefig("beta_2")
return_hist([samp[1] for samp in samps]).savefig("sigma")
return_hist([samp[2] for samp in samps]).savefig("rho")

#compute hpd
def hpd_calc(values, alpha):
    sort_vals = np.sort(values)
    best = 10**10
    lower = 0
    upper =0
    alpha_per = stats.percentileofscore(sort_vals, 100*alpha)
    for v in [x for x in sort_vals if x <= alpha_per]:
        #get percentile of v
        perc = stats.percentileofscore(sort_vals, v)
        #get corresponding endpoint
        try:
            end = np.percentile(sort_vals,perc +(1-alpha)*100)
            #return interval of minimal width that satisfies 
            if abs(v - end) < best:
                lower = v
                upper = end
        except:
            pass
    return lower,upper

#compute 5% intervals
print "95% HPD", hpd_calc([samp[2]for samp in samps],.05)
print "99% HPD", hpd_calc([samp[2]for samp in samps],.01)