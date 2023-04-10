

import numpy as np
import matplotlib.pyplot as plt
import ipdb
class NeuronAnalyzer: 
    
    '''
    Take cls embeddings from sentences with/without a particular attirbute and perform statistical analysis on the neurons.
    '''
    
    def __init__(self, activations1, activations2) -> None:
        self.activations1 = activations1
        self.activations2 = activations2
        
        self.neuron_rankings = None
        
        
    def plot_neuron_histograms(self, neuron_type, k=10, labels=["shuffled, unshuffled"], save_path="./figs"):
        
        if neuron_type == "top":
            neuron_indices = self.rank_neuron(neuron_type="top")[:k]
        if neuron_type == "bottom":
            neuron_indices = self.rank_neuron(neuron_type="bottom")[:k]
        
        # Create the figure and subplots
        fig, axs = plt.subplots((k+3)//4, 4, figsize=(30, 5*(k+3)//4))
    
        # Loop over each top neuron and plot its histograms on a subplot
        for i, neuron_idx in enumerate(neuron_indices):
            axs[i//4, i%4].hist(self.activations1[:, neuron_idx], bins=20, alpha=0.5, label=labels[0], color='blue')
            axs[i//4, i%4].hist(self.activations2[:, neuron_idx], bins=20, alpha=0.5, label=labels[1], color='red')
            axs[i//4, i%4].set_title('Histogram of activations for neuron {}'.format(neuron_idx))
            axs[i//4, i%4].set_xlabel('Activations')
            axs[i//4, i%4].set_ylabel('Frequency')
            axs[i//4, i%4].legend(loc='upper right')
    
        # Adjust the layout of the subplots
        fig.tight_layout()
    
        # Save the plot to a file and show it
        if neuron_type == "top":
            plt.savefig(f"{save_path}/histograms_top{k}_neurons_before_finetuning.png")
        if neuron_type == "bottom":
            plt.savefig(f"{save_path}/histograms_bottom{k}_neurons_before_finetuning.png")
        plt.show()

        

    # def compute_spearman_stat(self):
    #     self.neuron_rankings = []
    #     self._rank_neuron()
    
    def rank_neuron(self, metric, neuron_type="all", k=None, alpha=0.01):
        '''rank neurons based on the test statistic
        Args:
            metric: the test statistic to use
            k: the number of neurons to return, if k is None, return all activated/inactivated neurons
        '''
        neuron2stats_significant, neuron2stats_insignificant = self._compute_test_statistic(metric, alpha)
        
        # return a binary vector indicating whether a neuron is activated or not
        if neuron_type == "all":
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

    def _sort_neuron_by_statistic(dct, k=None, reverse=True):
        '''sort a dictionary by its values'''
        sorted_items = sorted(dct.items(), key=lambda item: item[1], reverse=reverse)
        # if k is None, return all items
        top_k_items = sorted_items[:k] if k is not None else sorted_items
        return {k: v for k, v in top_k_items}
