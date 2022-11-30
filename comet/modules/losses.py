import torch
import torch.nn as nn

class HeteroscedasticLoss(nn.Module):
    
    def forward(self, mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor):
        sigma = std**2
        log1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        mse = (target - mu)**2
        log2 = 0.5 * torch.log(sigma)
        return torch.sum(log1*mse+log2)


class HeteroscedasticLossv2(nn.Module):
    
    def forward(self, mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor):
        sigma = std
        log1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        mse = (target - mu)**2
        log2 = 0.5 * torch.log(sigma)
        return torch.sum(log1*mse+log2)

#Heteroscedastic inspired loss for error/uncertainty prediction
class HeteroApproxLoss(nn.Module):
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        sigma = pred**2
        l1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        l2 = 0.5 * torch.log(sigma)
        mse = target**2 
        #return torch.mean(0.5*pred**(-2)*(target**2)+(0.5*torch.log(pred**2)))
        return torch.sum(l1*mse+l2)

#Heteroscedastic inspired loss for error/uncertainty prediction
class HeteroApproxLossv2(nn.Module):
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        sigma = pred
        l1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        l2 = 0.5 * torch.log(sigma)
        mse = target**2 
        #return torch.mean(0.5*pred**(-2)*(target**2)+(0.5*torch.log(pred**2)))
        return torch.sum(l1*mse+l2)

class SquaredLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse = (target**2-pred**2)**2
        
        #return torch.mean(0.5*pred**(-2)*(target**2)+(0.5*torch.log(pred**2)))
        return torch.mean(mse)
 

class KLLoss(nn.Module):    
    #based on Daan's idea
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, target_mu: torch.Tensor, target_std: torch.Tensor):
       
        # Add fudge factor to variance to avoid large KL values
        #   (value of 1e-2 just turned out to work - 1e-3 already
        #   occasionally caused loss > 1000)
        std1 = target_std
        std2 = sigma
        mean1 = target_mu
        mean2 = mu
        
        kl = torch.log(torch.abs(std2)/torch.abs(std1)) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 0.5
        
        return kl.mean()


