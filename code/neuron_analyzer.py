

import numpy as np
import ipdb
class NeuronAnalyzer: 
    '''
    Take cls embeddings from sentences with/without a particular attirbute and perform statistical analysis on the activations for each neuron.
    '''
    
    def __init__(self, activations1, activations2) -> None:
        self.activations1 = activations1
        self.activations2 = activations2
        
        self.neuron_rankings = None
    
    def rank_neuron(self, metric, neuron_type, k=None, alpha=0.01):
        '''rank neurons based on the test statistic
        Args:
            metric: the test statistic to use
            k: the number of neurons to return, if k is None, return all activated/inactivated neurons
        '''
        neuron2stats_significant, neuron2stats_insignificant = self._compute_test_statistic(metric, alpha)
        
        # return a binary vector indicating whether a neuron is activated or not
        if neuron_type == "all" and k is None:
            res = np.zeros(self.activations1.shape[1])
            # ipdb.set_trace()
            res[list(neuron2stats_significant.keys())] = 1
            return res
        
        # return a dictionary of activated neurons and their test statistics
        if neuron_type == "top":
            return self._sort_neuron_by_statistic(neuron2stats_significant, k=k, reverse=True)
        
        # return a dictionary of inactivated neurons and their test statistics
        if neuron_type == "bottom":
            return self._sort_neuron_by_statistic(neuron2stats_insignificant, k=k, reverse=False)
        
    def compute_test_statistic_bonferroni(self, metric, alpha):
        num_neurons = self.activations1.shape[1]
        num_comparisons = self.activations2.shape[0]
        adj_alpha = alpha / num_comparisons

        res = np.zeros((num_neurons, num_comparisons))
        # loop over columns of A and corresponding columns of each 2-dimensional array in B and compare using KS test
        for i in range(num_neurons):
            for j in range(num_comparisons):
                _, p_val = metric(self.activations1[:,i],  self.activations2[j,:,i])
                if p_val <= adj_alpha:
                    res[i, j] = 1
        return np.sum(res, axis=1)

    def _compute_test_statistic(self, metric, alpha):
        '''compute the test statistic for each neuron'''
        num_of_neurons = self.activations1.shape[1]
        neuron2stats_significant = {}
        neuron2stats_insignificant = {}
        # ipdb.set_trace()
        for i in range(num_of_neurons):
            stat, p_value = metric(self.activations1[:, i], self.activations2[:, i])
            if p_value < alpha:
                neuron2stats_significant[i] = abs(stat)
            else: 
                neuron2stats_insignificant[i] = abs(stat)
        return neuron2stats_significant, neuron2stats_insignificant

    def _sort_neuron_by_statistic(self, dct, k, reverse):
        '''sort a dictionary by its values'''
        sorted_items = sorted(dct.items(), key=lambda item: item[1], reverse=reverse)
        # if k is None, return all items
        top_k_items = sorted_items[:k] if k is not None else sorted_items
        return {k: v for k, v in top_k_items}
