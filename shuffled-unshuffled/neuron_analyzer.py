
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
        
        
    def plot_neuron_histograms(self, neuron_type, k=10, save_path="./figs"):
        
        if neuron_type == "top":
            neuron_indices = self.rank_neuron(neuron_type="top")[:k]
        if neuron_type == "bottom":
            neuron_indices = self.rank_neuron(neuron_type="bottom")[:k]
        
        # Create the figure and subplots
        fig, axs = plt.subplots((k+3)//4, 4, figsize=(30, 5*(k+3)//4))
    
        # Loop over each top neuron and plot its histograms on a subplot
        for i, neuron_idx in enumerate(neuron_indices):
            axs[i//4, i%4].hist(self.pretained_activations[:, neuron_idx], bins=20, alpha=0.5, label="shuffled", color='blue')
            axs[i//4, i%4].hist(self.finetuned_activations[:, neuron_idx], bins=20, alpha=0.5, label="unshuffled", color='red')
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
    
    def rank_neuron(self, neuron_type, k=10, method="t_stat"): 
        if method == "t_stat":
            t_stats = self._compute_t_stat()
            # Get the indices of the sorted array in ascending or descending order depending on neuron type whether top or bottom
            if neuron_type == "top":
                sorted_neurons_indices = np.argsort(t_stats)[::-1]
            if neuron_type == "bottom":
                sorted_neurons_indices = np.argsort(t_stats)
            
        return list(sorted_neurons_indices)[:k]


    def _compute_t_stat(self, alpha=0.05):
        # num_of_neurons = self.pretained_activations.shape[1]
        t_stats, p_vals = ttest_ind(self.pretained_activations, self.finetuned_activations, axis=0)
        significant_neurons = np.where(p_vals < alpha)[0]
        # insignificant_neurons = np.where(p_vals >= alpha)[0]
        # ipdb.set_trace()
        # todo
        # return np.concatenate((t_stats[significant_neurons], t_stats[insignificant_neurons]), axis=0)
        return t_stats[significant_neurons]