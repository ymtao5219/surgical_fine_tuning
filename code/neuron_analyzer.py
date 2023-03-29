
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import ipdb
class NeuronAnalyzer: 
    
    '''
    Take cls embeddings from pretrained and finetuned models and perform statistical analysis on the neurons.
    '''
    
    def __init__(self, pretained_activations, finetuned_activations) -> None:
        self.pretained_activations = pretained_activations
        self.finetuned_activations = finetuned_activations
        
        self.neuron_rankings = None
        
    
    def plot_neuron_histograms(self, top_k=10, save_path="./figs"):
        neuron_indices = self.rank_neuron()[:top_k]
        for top_i, neuron_idx in enumerate(neuron_indices): 
            
            plt.hist(self.pretained_activations[:, neuron_idx], bins=20, alpha=0.5, label="pretrained", color='blue')
            plt.hist(self.finetuned_activations[:, neuron_idx], bins=20, alpha=0.5, label="finetuned", color='red')
            plt.title('Histogram of activations for neuron {}'.format(neuron_idx))
            plt.xlabel('Actviations')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')
            plt.savefig(f"{save_path}/histograms_top{top_i}_neuron_{neuron_idx}.png")
            plt.clf()

    # def compute_spearman_stat(self):
    #     self.neuron_rankings = []
    #     self._rank_neuron()
    
    def rank_neuron(self, top_k=10, method="t_stat"): 
        if method == "t_stat":
            t_stats = self._compute_t_stat()
            # Get the indices of the sorted array in descending order
            sorted_neurons_indices = np.argsort(t_stats)[::-1]
        return list(sorted_neurons_indices)[:top_k]
        
    def _compute_t_stat(self, alpha=0.05):
        # num_of_neurons = self.pretained_activations.shape[1]
        t_stats, p_vals = ttest_ind(self.pretained_activations, self.finetuned_activations, axis=0)
        significant_neurons = np.where(p_vals < alpha)[0]
        # ipdb.set_trace()
        return t_stats[significant_neurons]