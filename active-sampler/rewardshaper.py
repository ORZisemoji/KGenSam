import numpy as np

class RewardShaper(object):
    def __init__(self,args):
        self.shaping = args.shaping
        self.entcoef = args.entcoef
        self.gamma = 0.5
        self.rate = 0.05
        self.alpha = args.frweight # to make tune the ratio of finalreward and midreward


    def reshape(self,rewards_all,finalrewards_all,logprobs):
        self.batchsize=len(finalrewards_all)
        if "0" in self.shaping and "1" in self.shaping:
            raise ValueError("arguments invalid")
        self.hashistorymean,self.hashistoryvar = False,False
        if "3" in self.shaping:
            self.historymean = np.zeros((1000,self.batchsize))
        if "4" in self.shaping:
            self.histrvar = np.zeros((1000,self.batchsize))

        rewards_sub, finalrewards = self._roughProcess(rewards_all,finalrewards_all)
        self.componentRatio(rewards_sub,finalrewards,logprobs)
        rewards = np.zeros_like(rewards_sub)
        if "0" in self.shaping:
            rewards += rewards_sub
        if "1" in self.shaping:
            for i in range(rewards_sub.shape[0]-1,0,-1):
                rewards_sub[i-1] += self.gamma*rewards_sub[i]
            rewards += rewards_sub
        if "2" in self.shaping: #2
            rewards=rewards[:,] +finalrewards*self.alpha

        if "3" in self.shaping:
            if not self.hashistorymean:
                self.historymean[:rewards_sub.shape[0], :] += rewards.mean(1,keepdims=True)
                self.hashistorymean = True
            else:
                self.historymean[:rewards_sub.shape[0], :] = self.historymean[:rewards_sub.shape[0], :] * (1 - self.rate) + self.rate * rewards.mean(1,keepdims=True)
            rewards = rewards - self.historymean[:rewards.shape[0], :]

        if "4" in self.shaping:
            if not self.hashistoryvar:
                self.histrvar[:rewards_sub.shape[0], :] += (rewards**2).mean(1,keepdims=True)
                self.hashistoryvar = True
            else:
                self.histrvar[:rewards_sub.shape[0], :] = self.histrvar[:rewards.shape[0], :] * (
                            1 - self.rate) + self.rate * (rewards**2).mean(1,keepdims=True)
            rewards = rewards/np.power(self.histrvar[:rewards.shape[0], :],0.5)
        return rewards

    def _roughProcess(self,rewards_all,finalrewards_all):
        rewards = np.array(rewards_all)
        finalrewards = np.array(finalrewards_all)
        # print('rewards_all:{}'.format(rewards_all))
        # print('list(zip(*rewards_all)):{}'.format(list(zip(*rewards_all))))
        # print('rewards:{}'.format(rewards))
        # print('finalrewards:{}'.format(finalrewards))
        rewards_sub = rewards[1:]-rewards[:-1]
        return rewards_sub,finalrewards

    def componentRatio(self,rewards_sub,finalrewards_all,logprobs):
        r_mean = np.mean(np.abs(rewards_sub))
        f_mean = np.mean(finalrewards_all)
        lp_mean = np.mean(np.abs(logprobs))
        f_ratio = f_mean/r_mean*self.alpha
        lp_ratio = lp_mean/r_mean*self.entcoef
        print("rmean {:.4f},fratio {:.2f}x{:.4f}={:.3f}, "
                    "lpratio {:.1f}x{:.5f}={:.3f}".format(r_mean,
                                                           f_mean/r_mean,self.alpha,f_ratio,
                                                           lp_mean/r_mean,self.entcoef,lp_ratio))