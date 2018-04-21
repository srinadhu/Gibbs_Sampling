import numpy as np
import pickle
import random
import copy
import matplotlib.pyplot as plt

from collections import defaultdict


def Bayes_Net_Prob(marginals, joints, trijoints, test_samples):
	"""
	Returns the probability of a sample using the bayes net simplification from the network we already got
        """

	num_test = len(test_samples)
	bayes_net_prob = np.ones(num_test)

	for i in range(num_test):
		s = test_samples[i]

		try:
			#feature 0
			bayes_net_prob[i] *= marginals[0][s[0]]
			#feature 1 
			bayes_net_prob[i] *= marginals[1][s[1]]
			#feature 2
			bayes_net_prob[i] *= marginals[2][s[2]]
			#feature 3
			bayes_net_prob[i] *= (joints[2, 3][s[2],s[3]] / marginals[2][s[2]])		
			#feature 4
			bayes_net_prob[i] *= (joints[0,4][s[0],s[4]] / marginals[0][s[0]] )
			#feature 5
			bayes_net_prob[i] *= (trijoints[1,2,5][s[1],s[2],s[5]] / joints[1,2][s[1],s[2]])
			#feature 6
			bayes_net_prob[i] *= (joints[4,6][s[4],s[6]] / marginals[4][s[4]])
			#feature 7
			bayes_net_prob[i] *= marginals[7][s[7]]
			#feature 8
			bayes_net_prob[i] *= (joints[6,8][s[6],s[8]] / marginals[6][s[6]] )
			#feature 9
			bayes_net_prob[i] *= (joints[0, 9][s[0],s[9]] / marginals[0][s[0]])		
			#feature 10
			bayes_net_prob[i] *= (joints[0, 10][s[0],s[10]] / marginals[0][s[0]] )
			#feature 11
			bayes_net_prob[i] *= (trijoints[0,5, 11][s[0],s[5],s[11]] / joints[0,5][s[0],s[5]])
			#feature 12
			bayes_net_prob[i] *= (trijoints[2,7, 12][s[2],s[7],s[12]] / joints[2,7][s[2],s[7]])
			#feature 13
			bayes_net_prob[i] *= (joints[6,13][s[6],s[13]] / marginals[6][s[6]])
		
		except:
			bayes_net_prob[i] = 0.0

	return bayes_net_prob

#Edges, Marginals, Joints are useful
def Gibbs_Sampling(marginals, joints, trijoints, bayes_net_prob, test_samples, burn_in = 10, num_samples = 10000, num_features = 14,verbose = True):
    """
	Samples the given number of samples and returns the mean squared error from Bayes Net and Sampling for Test dataset

	i/p: marginals: will have marginals for all 14 features
	     joints: will have probabilities for all joints of different feature pair values P(feature1_value, feature2_value)
	     trijoints: will have probabilities for all triplets of different feature triplet values P(feature1_value, feature2_value,
		            feature3_value)
	     bayes_net_prob: Probability of all test samples from Bayes net
	     test_samples: all the test samples or test dataset
	     burn_in: Number of samples rejecting for every sample generated since two generated one after another aren't truly independent
	     num_samples: number of samples to generate
	     num_features: total number of features
	     verbose: To print which iteration is running if needed

	o/p:
	     returns the mean squared error b/w bayes_net_prob and sampling probability for test_samples
    """
    samples = [] #all the samples
    sample_init = sample_intilization(marginals, joints, trijoints, num_features = 14)
    samples.append(sample_init)
    sample = copy.deepcopy(sample_init)
   
    for i in range(num_samples): #get those many samples

        for j in range(burn_in): #wait till burn_in samples
            k = random.randrange(num_features)
            sample_init = sample_next_feature(marginals, joints, trijoints,  sample, k)
	    sample = copy.deepcopy(sample_init)

        #generate a sample and add it
	feature_index = random.randrange(num_features)
   	sample_init = sample_next_feature(marginals, joints, trijoints, sample, feature_num = feature_index)
	sample = copy.deepcopy(sample_init)
	samples.append(sample)	

        if (i%100 == 0 and verbose):
            print "iteration " + str(i) + " done."

    test_prob = Test_Sample_Prob(test_samples, samples)
    error_estimate  = Error_Sampling(bayes_net_prob, test_prob)

    return error_estimate


def sample_next_feature(marginals, joints, trijoints,  sample_init, feature_num, num_features = 14):
	""" Samples the next sample by conditioning on all except the 'feature_num' and generate a value for it
	     
	    i/p: marginals: will have marginals for all 14 features
		 joints: will have probabilities for all joints of different feature pair values P(feature1_value, feature2_value)
		 trijoints: will have probabilities for all triplets of different feature triplet values P(feature1_value, feature2_value,
		            feature3_value)

		 sample_init: The values for all the features before sampling
		 feature_num: which feature to sample in this sampling
		 num_features: total number of features

	    o/p: returns a sample
	"""
	
	sample_feature_value = None
	possible_values = marginals[feature_num].keys() #all the possible values
	prob = np.ones(len(possible_values))

	for (i,value) in enumerate(possible_values):
		sample_init[feature_num] = value
		prob[i] = Bayes_Net_Prob(marginals, joints, trijoints, [sample_init])

	prob /= np.sum(prob)

	[sample_feature_value] = np.random.choice (possible_values, 1, p = list(prob)) #sample now for one.	

	sample_init[feature_num] = sample_feature_value

	return sample_init
	

def Test_Sample_Prob(test_samples, samples):
	""" Get the probability of test samples from sampling. """

	num_test_samples = len(test_samples) #number of test samples
	num_samples = float(len(samples)) #total samples sampled from gibbs

	test_prob = np.zeros(num_test_samples)

	for i in range(num_test_samples): #get each sample probability
		test_prob[i] = samples.count(test_samples[i]) / num_samples 
	
	return test_prob #returns the probability

def Error_Sampling(bayes_net_prob, test_prob):
	""" Gives sum of squared error for Bayes Net probabilities and Gibbs Sampling Probabilities"""

	bayes_net = np.array(bayes_net_prob)
	test = np.array(test_prob)

	error = np.mean(np.square(bayes_net - test)) # Error b/w sampling and bayes net

	return error

def sample_intilization(marginals, joints, trijoints, num_features = 14):
	""" Marginals will have the probability of
	    choosing a particular value for a specific
	    random variable, using it to initialize
            a sample and returns it

	    i/p: marginals: probability distribution of
			    each random variable stored
			    in dictionary i.e., dictionary
			    within a dictionary

	    o/p: returns a sample"""

	sample = []

	while(1):
		for i in range(14):
			feature_dict = marginals[i] #specific dictionary for the ith feature
			[feature_i_value] = np.random.choice(feature_dict.keys(), 1, p = feature_dict.values()) #pick a value based on pdf
			sample.append(feature_i_value)

		prob = Bayes_Net_Prob(marginals, joints, trijoints, [sample])
		if (prob == 0):
			sample = []
			continue
		else:
			break
	return sample


fl = open('Bayes_Net', 'rb')
marginals = pickle.load(fl)
joints = pickle.load(fl)
trijoints = pickle.load(fl)
test_samples = pickle.load(fl)
fl.close()

bayes_net_prob = Bayes_Net_Prob(marginals, joints, trijoints, test_samples)

errors = []
nums = []
for i in range(1, 11):
	error = 0.0
	for j in range(i):
		error += Gibbs_Sampling(marginals, joints, trijoints, bayes_net_prob, test_samples, burn_in = 10, num_samples = 20000, num_features = 14, verbose = False)
	error /= i
	errors.append(error)
	nums.append(i)
	print "sampling ended " + str(i)

#get what plot you need for this.
plt.title(" Squared Error vs No of runs ")
plt.xlabel("Number of runs of Gibbs Sampler")
plt.ylabel("Squared error b/w BNet & Sampling probability")
plt.plot(nums, errors)
plt.savefig("error.png")
