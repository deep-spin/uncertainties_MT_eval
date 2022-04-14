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
 



