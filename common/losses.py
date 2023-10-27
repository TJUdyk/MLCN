
import torch
import torch.nn as nn
import torch.nn.functional as F

def contrast_distill(f1, f2):
    """
    Contrastive Distillation
    """
    f1 = F.normalize(f1, dim=1, p=2)
    f2 = F.normalize(f2, dim=1, p=2)
    loss = 2 - 2 * (f1 * f2).sum(dim=-1)
    return loss.mean()


class DistillKL(nn.Module):
    """
    KL divergence for distillation
    """
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (based on https://github.com/HobbitLong/SupContrast)
    """
    def __init__(self, args,temperature=None):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.args = args

    def _compute_logits(self, features_a, features_b, attention=None):

        
        # global similarity
        if features_a.dim() == 2:
            features_a = F.normalize(features_a, dim=1, p=2)
            features_b = F.normalize(features_b, dim=1, p=2)
            contrast = torch.matmul(features_a, features_b.T)

        # spatial similarity
        elif features_a.dim() == 4:
            contrast = attention(features_a, features_b)

        else:
            raise ValueError
        
        # note here we use inverse temp
        contrast = contrast * self.temperature 
        '''
          # projection of features a  torch.Size([75, 25, 640])
        spt = features_b  # torch.Size([75, 640, 5, 5])
        way , c_spt, h_spt , w_spt = spt.size()
        spt_attended = spt
        spt_attended = spt_attended.view(way , c_spt , h_spt*w_spt)
        spt_attended = self.gaussian_normalize(spt_attended, dim=-1)# torch.Size([5, 640, 25])
        atten_score_spt = F.softmax(spt_attended/self.args.temperature_attn, dim =-1) # torch.Size([5, 640, 25])
        #atten_score_spt = spt_attended.view(way , c_spt , h_spt, w_spt) # torch.Size([5, 640, 5, 5])
        atten_score_spt = atten_score_spt.mean(dim=-1) # torch.Size([5, 640, 5])
        spt = spt.mean(dim=[-1 , -2])
        atten_score_spt = atten_score_spt * spt # torch.Size([5, 640])
        #atten_score_spt = atten_score_spt + spt

        qry = features_a
        num_qry, c_qry, h_qry , w_qry = qry.size() # torch.Size([75, 64])
        qry_attended = qry
        qry_attended = qry_attended.view(num_qry , c_qry , h_qry*w_qry) # torch.Size([75, 640, 25])
        qry_attended = self.gaussian_normalize(qry_attended, dim=-1)# torch.Size([75, 640, 25])
        atten_score_qry = F.softmax(qry_attended/self.args.temperature_attn, dim =-1)
        atten_score_qry = atten_score_qry.mean(dim=-1)
        #atten_score_qry = atten_score_qry.view(num_qry , c_qry , h_qry,w_qry)
        qry = qry.mean(dim =[-1 , -2])
        atten_score_qry = atten_score_qry * qry # torch.Size([75, 640])
        #atten_score_qry = atten_score_qry + qry
        
        spt_attended = atten_score_spt.unsqueeze(0).repeat(num_qry, 1, 1)  # torch.Size([75, 5, 640, 5, 5])
        qry_attended = atten_score_qry.unsqueeze(1).repeat(1, way, 1) # torch.Size([75, 5, 640, 5, 5])


        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        similarity_matrix  = F.cosine_similarity(qry_attended, spt_attended, dim=-1)
        '''
        return contrast
    
    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x
    
    def forward(self, features_a, features_b=None, labels=None, attention=None):
        # features_a :torch.Size([75, 640, 5, 5])   features_b: torch.Size([5, 640, 5, 5])
        device = (torch.device('cuda') if features_a.is_cuda else torch.device('cpu')) # torch.Size([5, 640, 5, 5])
        num_features, num_labels = features_a.shape[0], labels.shape[0]
        #num_features_b = features_b.shape[0]

        # using only the current features in a given batch
        if features_b is None:
            features_b = features_a
            # mask to remove self contrasting
            logits_mask = (1. - torch.eye(num_features)).to(device)
        else:
            # contrasting different features (a & b), no need to mask the diagonal
            logits_mask = torch.ones(num_features, num_features).to(device)  # torch.Size([75, 75])
        
        # mask to only maintain positives
        if labels is None:
            # standard self supervised case
            mask = torch.eye(num_labels, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)  # torch.Size([75, 1])
            mask = torch.eq(labels, labels.T).float().to(device)

        # replicate the mask since the labels are just for N examples
        if num_features != num_labels:
            assert num_labels * 2 == num_features
            mask = mask.repeat(2, 2)

        # compute logits   features_a:torch.Size([75, 640, 5, 5]) features_b:torch.Size([75, 640, 5, 5])
        contrast = self._compute_logits(features_a, features_b, attention) # torch.Size([75, 75])

        # remove self contrasting # mask : torch.Size([75, 75]) 
        mask = mask * logits_mask

        # normalization over number of positives
        normalization = mask.sum(1)  # torch.Size([75])
        normalization[normalization == 0] = 1.

        # for stability
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True) # torch.Size([75, 1])
        logits = contrast - logits_max.detach() # torch.Size([75, 75])
        exp_logits = torch.exp(logits) # torch.Size([75, 5])

        exp_logits = exp_logits * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / normalization
        loss = -mean_log_prob_pos.mean()

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
