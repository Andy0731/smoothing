import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from train_utils import add_fnoise, add_fnoise_chn
import torchvision


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, 
        num_classes: int, 
        sigma: float, 
        use_amp: bool = False,
        favg: int = 0,
        avgn_loc: str = None, 
        avgn_num: int = 1,
        fnoise_sd: float = 0.0,
        get_samples = False,
        nconv: int = 0,
        hug: bool = False,
        resize_after_noise: int = 0,
        diffusion_model = None,
        args = None
        ):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.use_amp = use_amp
        self.favg = favg
        self.avgn_loc = avgn_loc
        self.avgn_num = avgn_num
        self.fnoise_sd = fnoise_sd
        self.get_samples = get_samples
        self.nconv = nconv
        self.hug = hug
        self.resize_after_noise = resize_after_noise
        self.diffusion_model = diffusion_model
        self.args = args

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> tuple:
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        if self.get_samples:
            counts_selection, _, _ = self._sample_noise(x, n0, batch_size)
        else:
            counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        if self.get_samples:
            counts_estimation, imgs, preds = self._sample_noise(x, n, batch_size)
        else:
            counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if self.get_samples:
            if pABar < 0.5:
                return Smooth.ABSTAIN, 0.0, imgs, preds
            else:
                radius = self.sigma * norm.ppf(pABar)
                return cAHat, radius, imgs, preds
        else:                        
            if pABar < 0.5:
                return Smooth.ABSTAIN, 0.0
            else:
                radius = self.sigma * norm.ppf(pABar)
                return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1)) #b,c,h,w

                # noise = torch.randn_like(batch, device='cuda') * self.sigma
                # batch += noise
                # if self.resize_after_noise:
                #     batch = torchvision.transforms.functional.resize(batch, self.resize_after_noise)

                if hasattr(self.args, 'diffusion') and self.args.diffusion:
                    acc_noise = self.args.accurate_noise if hasattr(self.args, 'accurate_noise') else 0
                    batch = self.diffusion_model(batch, self.args.t, acc_noise, self.args.noise_sd)
                else:
                    batch = batch + torch.randn_like(batch, device='cuda') * self.sigma
                if hasattr(self.args, 'resize_after_noise'):
                    # inputs = torchvision.transforms.functional.resize(inputs, args.resize_after_noise)
                    batch = torch.nn.functional.interpolate(batch, self.args.resize_after_noise, mode='bicubic')

                # expand (x + gnoise) with k fnoise 
                if self.favg:
                    batch = add_fnoise(batch, self.fnoise_sd, self.avgn_num)
                elif self.nconv:
                    batch = add_fnoise_chn(batch, self.fnoise_sd, self.avgn_num)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.base_classifier(batch)
                    if self.hug:
                        outputs = outputs.logits
                    predictions = outputs.argmax(1)

                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            if self.get_samples:
                return counts, batch, predictions
            else:
                return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
