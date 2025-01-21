import numpy as np
import scipy.stats as stats
from toyModel.utils import ternary_search_max, ternary_search_min


class StatisticalModel:

    def __init__(self, K, x_test, y_test, mc_samples=100):
        """
        Initialize the statistical model.

        Parameters:
        K (int): Number of intervals to divide [0, 1] into.
        Q (int): Number of bins of the data-generating distribution.
        """
        if K <= 0:
            raise ValueError("K must be greater than 0")

        self.mc_samples = mc_samples
        self.K = K
        self.bin_counts = None
        self.expected_bin_counts = None

        # Generate 10,000 samples and compute expected counts
        self.test_x_sample = x_test
        self.test_y_sample = y_test
        self.expected_bin_counts = self.count_observations_per_bin(self.test_x_sample, self.test_y_sample)

    @staticmethod
    def generate_sample(n, K, noise=0.2, random_seed=None):
        """
        Generate a sample of n values from a uniform distribution between 0 and 1.
        Also generate corresponding y values from a binomial distribution,
        where the binomial parameter depends on the interval of x.

        Parameters:
        n (int): Number of samples to generate.
        K (int): Number of intervals to divide [0, 1] into.
        noise (float): Noise parameter to determine binomial probabilities (default is 0.2).

        Returns:
        tuple: Tuple containing x_sample (numpy.ndarray) and y_sample (numpy.ndarray).
        """
        if K <= 0:
            raise ValueError("K must be greater than 0")

        if random_seed is not None:
            np.random.seed(random_seed)

        x_sample = np.random.uniform(0, 1, n)
        #x_sample = np.linspace(0,1, n)
        interval_indices = (x_sample // (1.0 / K)).astype(int)
        interval_indices[interval_indices == K] = K - 1

        y_sample = np.random.binomial(1, np.where(interval_indices % 2 == 0, noise, 1 - noise))

        return x_sample, y_sample

    def generate_average_sample(n, K, noise=0.2):
        """
        Generate an 'average' dataset where labels are assigned deterministically
        based on the interval probabilities.

        Parameters:
        n (int): Number of samples to generate.
        K (int): Number of intervals to divide [0, 1] into.
        noise (float): Noise parameter (default is 0.2).

        Returns:
        tuple: Tuple containing x_sample (numpy.ndarray) and y_sample (numpy.ndarray).
        """
        if K <= 0:
            raise ValueError("K must be greater than 0")
        if not (0 <= noise <= 1):
            raise ValueError("Noise must be between 0 and 1")

        # Generate x values
        x_sample = np.linspace(0, 1, n, endpoint=False)

        # Map x_sample to interval indices
        interval_indices = (x_sample * K).astype(int)

        # Determine probabilities for each interval
        probabilities = np.where(interval_indices % 2 == 0, noise, 1 - noise)

        # Assign labels based on probabilities
        y_sample = (probabilities >= 0.5).astype(int)

        return x_sample, y_sample

    def essentialInf(self):
        return np.min(self.model_test_losses)
    def setAlphaPrior(self, alpha):
        self.alpha_prior = alpha

        if (alpha == 0):
            def auxiliar_function(alpha):
                self.alpha_prior = alpha
                self.compute_beta_distributions()
                return self.logMLL()

            min_alpha = 0.0
            max_alpha = 50

            _, alpha_star = ternary_search_min(auxiliar_function,
                                                       min_alpha,
                                                       max_alpha,
                                                       epsilon=0.1)
            self.alpha_prior = alpha_star
            self.compute_beta_distributions()

    def count_observations_per_bin(self, x_sample, y_sample):
        """
        Count the number of observations where y=1 and y=0 for each of the K bins.

        Parameters:
        x_sample (numpy.ndarray): Array of x values.
        y_sample (numpy.ndarray): Array of y values.

        Returns:
        numpy.ndarray: A numpy array of shape (K, 2), where each row contains (count_y0, count_y1) for each bin.
        """
        interval_indices = (x_sample // (1.0 / self.K)).astype(int)
        interval_indices[interval_indices == self.K] = self.K - 1
        bin_counts_y0 = np.bincount(interval_indices[y_sample == 0], minlength=self.K)
        bin_counts_y1 = np.bincount(interval_indices[y_sample == 1], minlength=self.K)

        bin_counts = np.column_stack((bin_counts_y0, bin_counts_y1))
        return bin_counts

    def set_training_data(self, x_sample, y_sample):
        """
        Set the training data by calculating and storing the counts of observations per bin.

        Parameters:
        x_sample (numpy.ndarray): Array of x values.
        y_sample (numpy.ndarray): Array of y values.
        """
        self.train_x_sample = x_sample
        self.train_y_sample = y_sample
        self.bin_counts = self.count_observations_per_bin(x_sample, y_sample)

    def compute_beta_distributions(self):
        """
        Compute the beta distributions for each bin representing the posterior.
        Each beta distribution is defined by the counts of the bin plus prior counts.

        Parameters:
        alpha_prior (float): Prior count to be added to both y=0 and y=1 counts (default is 1).
        """
        if self.bin_counts is None:
            raise ValueError("Training data has not been set. Please call set_training_data first.")

        self.posterior_beta_distributions = np.empty(self.K, dtype=object)
        for bin_idx in range(self.K):
            count_y0, count_y1 = self.bin_counts[bin_idx]
            a = count_y1 + self.alpha_prior
            b = count_y0 + self.alpha_prior
            self.posterior_beta_distributions[bin_idx] = stats.beta(a, b)

        self.sampled_models = [self.sample_posterior_probs() for _ in range(self.mc_samples)]
        self.model_test_losses = [self.vecLogLoss(self.test_x_sample, self.test_y_sample, probs) for probs in self.sampled_models]
        self.model_train_losses = [self.vecLogLoss(self.train_x_sample, self.train_y_sample, probs) for probs in self.sampled_models]

    def KLPosteriorPrior(self):
        """
        Calculate the sum of the KL divergence between the beta posterior produced by compute_beta_distributions
        and a prior beta distribution defined by an alpha_prior parameter.


        Returns:
        float: The sum of KL divergences for all bins.
        """
        if self.bin_counts is None:
            raise ValueError("Training data has not been set. Please call set_training_data first.")

        kl_sum = 0.0
        for bin_idx in range(self.K):
            beta_dist = self.posterior_beta_distributions[bin_idx]
            a_post = beta_dist.args[0]
            b_post = beta_dist.args[1]
            a_prior = self.alpha_prior
            b_prior = self.alpha_prior

            kl_divergence = (stats.beta(a_post, b_post).mean() * (a_post - a_prior) +
                             (1 - stats.beta(a_post, b_post).mean()) * (b_post - b_prior))
            kl_sum += kl_divergence

        return kl_sum

    def sample_posterior_probs(self):
        """
        Sample probabilities from the posterior beta distributions.

        Parameters:
        num_samples (int): Number of samples to draw from each beta distribution.

        Returns:
        numpy.ndarray: Array of sampled probabilities.
        """

        return np.array([beta_dist.rvs() for beta_dist in self.posterior_beta_distributions])



    def vecLogLoss(self, x_sample, y_sample, probs):
        """
        Compute the log loss given counts and probabilities.

        Parameters:
        counts_y0 (numpy.ndarray): Array of counts where y=0 for each bin.
        counts_y1 (numpy.ndarray): Array of counts where y=1 for each bin.
        probs (numpy.ndarray): Array of probabilities for each bin.

        Returns:
        float: The log loss value.
        """
        interval_indices = (x_sample // (1.0 / self.K)).astype(int)
        interval_indices[interval_indices == self.K] = self.K - 1
        p_y1 = probs[interval_indices]

        log_loss = -y_sample*np.log(p_y1+0.000001) - (1-y_sample)*np.log(1 - p_y1 + 0.000001)

        return log_loss

    def _expectationTest(self, function):
        """
        Compute the log loss from multiple samples of the posterior beta distributions.
        This is a Monte Carlo approximation of the expectation over the posterior beta distributions.

        Parameters:
        counts_y0 (np.ndarray): Array of counts where y=0 for each bin.
        counts_y1 (np.ndarray): Array of counts where y=1 for each bin.
        num_samples (int): Number of samples to draw from each beta distribution for Monte Carlo approximation.

        Returns:
        float: The average log loss value.
        """
        if self.posterior_beta_distributions is None:
            raise ValueError(
                "Posterior beta distributions have not been computed. Please call compute_beta_distributions first.")

        val = 0.0
        for losses in self.model_test_losses:
            val += function(losses)

        val = val / len(self.model_test_losses)
        return val

    def _expectationTrain(self, function):
        """
        Compute the log loss from multiple samples of the posterior beta distributions.
        This is a Monte Carlo approximation of the expectation over the posterior beta distributions.

        Parameters:
        counts_y0 (np.ndarray): Array of counts where y=0 for each bin.
        counts_y1 (np.ndarray): Array of counts where y=1 for each bin.
        num_samples (int): Number of samples to draw from each beta distribution for Monte Carlo approximation.

        Returns:
        float: The average log loss value.
        """
        if self.posterior_beta_distributions is None:
            raise ValueError(
                "Posterior beta distributions have not been computed. Please call compute_beta_distributions first.")

        val = 0.0
        for losses in self.model_train_losses:
            val += function(losses)

        val = val / len(self.model_train_losses)
        return val

    def empiricalGibbsLoss(self):
        """
        Compute the log loss of the bin_counts from multiple samples of the posterior beta distributions.
        This is a Monte Carlo approximation of the expectation over the posterior beta distributions.

        Parameters:
        num_samples (int): Number of samples to draw from each beta distribution for Monte Carlo approximation.

        Returns:
        float: The average log loss value.
        """
        def auxiliar_function(train_loss):
            return np.mean(train_loss)

        return self._expectationTrain(auxiliar_function)
    def expectedGibbsLoss(self):
        """
        Compute the log loss of the expected_bin_counts from multiple samples of the posterior beta distributions.
        This is a Monte Carlo approximation of the expectation over the posterior beta distributions.

        Parameters:
        num_samples (int): Number of samples to draw from each beta distribution for Monte Carlo approximation.

        Returns:
        float: The average log loss value.
        """
        def auxiliar_function(losses):
            return np.mean(losses)

        return self._expectationTest(auxiliar_function)

    def expectedBMALoss(self):
        """
        Compute the log loss of the expected_bin_counts from multiple samples of the posterior beta distributions.
        This is a Monte Carlo approximation of the expectation over the posterior beta distributions.

        Parameters:
        num_samples (int): Number of samples to draw from each beta distribution for Monte Carlo approximation.

        Returns:
        float: The average log loss value.
        """
        def auxiliar_function(losses):
            return np.exp(-losses)

        return np.mean(-np.log(self._expectationTest(auxiliar_function)))

    def expectedGibbsVarLoss(self):
        """
        Compute the log loss of the expected_bin_counts from multiple samples of the posterior beta distributions.
        This is a Monte Carlo approximation of the expectation over the posterior beta distributions.

        Parameters:
        num_samples (int): Number of samples to draw from each beta distribution for Monte Carlo approximation.

        Returns:
        float: The average log loss value.
        """
        def auxiliar_function(losses):
            return np.var(losses)

        return self._expectationTest(auxiliar_function)

    def cummulant(self, lambda_, test_losses):
        L = np.mean(test_losses)

        all = -lambda_*test_losses

        max_all = np.max(all)
        log_aveg_exp = max_all + np.log(np.sum(np.exp(all - max_all))) - np.log(all.shape[0])

        return lambda_*L + log_aveg_exp




    def expectedCumulant(self, lambda_):
        """
        Compute the expected cumulant using Monte Carlo approximation over samples from the posterior.

        Parameters:
        bin_idx (int): Index of the bin.
        lambda_ (float): Parameter lambda for the cumulant.
        num_samples (int): Number of samples to draw from each beta distribution for Monte Carlo approximation (default is 100).

        Returns:
        float: The expected cumulant value.
        """

        def auxiliar_function(test_losses):
            return self.cummulant(lambda_,test_losses)

        return self._expectationTest(auxiliar_function)



    def rate(self, a):
        def auxiliar_function(lamb):
            cummulant = self.expectedCumulant(lamb)
            return lamb * a - cummulant

        if a < 0:
            min_lambda = -10000
            max_lambda = 0
        else:
            min_lambda = 0
            max_lambda = 10000

        opt_func, lambda_star = ternary_search_max(auxiliar_function,
                                                   min_lambda,
                                                   max_lambda,
                                                   epsilon=0.01)

        return opt_func, lambda_star

    def inv_rate(self, a):
        def auxiliar_function(lamb):
            cummulant = self.expectedCumulant(lamb)
            return (cummulant + a) / lamb

        if a < 0:
            min_lambda = -10000
            max_lambda = 0
        else:
            min_lambda = 0
            max_lambda = 10000

        opt_func, lambda_star = ternary_search_min(auxiliar_function,
                                                   min_lambda,
                                                   max_lambda,
                                                   epsilon=0.01)

        return opt_func, lambda_star


    def logMLL(self):
            return self.empiricalGibbsLoss() + self.KLPosteriorPrior()/self.train_x_sample.shape[0]


    def gapBayesGibbs(self):
        return self.expectedGibbsLoss()-self.expectedBMALoss()
    def PAC_Chernoff(self):
        inv_rate, lambda_star = self.inv_rate((self.KLPosteriorPrior()+np.log(self.train_x_sample.shape[0]/0.05))/(self.train_x_sample.shape[0]-1))
        return self.empiricalGibbsLoss() + inv_rate, lambda_star

    def PAC_ChernoffFull(self):
        return self.PAC_Chernoff() - self.gapBayesGibbs()
    def PAC_ChernoffSubGaussian(self):
        complexity = (self.KLPosteriorPrior()+np.log(self.train_x_sample.shape[0]/0.05))/(self.train_x_sample.shape[0]-1)
        complexity  = complexity*self.expectedGibbsVarLoss()
        return self.empiricalGibbsLoss() + np.sqrt(complexity)

    def PAC_ChernoffSubGaussianFull(self):
        return self.PAC_ChernoffSubGaussian() - self.gapBayesGibbs()

