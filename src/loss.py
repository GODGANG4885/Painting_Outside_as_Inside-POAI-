import torch
import torch.nn as nn
import torchvision.models as models


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss


# class HistogramLoss(nn.Module):
#     r"""
#     Histogram Loss, VGG-based
#     https://arxiv.org/pdf/1611.00822.pdf
#     https://github.com/valerystrizh/pytorch-histogram-loss/blob/master/losses.py
#     """
#     def __init__(self, num_steps, cuda=True):
#         super(HistogramLoss, self).__init__()
#         self.step = 2 / (num_steps - 1)
#         self.eps = 1 / num_steps
#         self.cuda = cuda
#         self.t = torch.arange(-1, 1+self.step, self.step).view(-1, 1)
#         self.tsize = self.t.size()[0]
#         if self.cuda:
#             self.t = self.t.cuda()
        
#     def forward(self, features, classes):
#         def histogram(inds, size):
#             s_repeat_ = s_repeat.clone()
#             indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) & (s_repeat_floor - (self.t - self.step) < self.eps) & inds
#             assert indsa.nonzero().size()[0] == size, ('Another number of bins should be used')
#             zeros = torch.zeros((1, indsa.size()[1])).byte()
#             if self.cuda:
#                 zeros = zeros.cuda()
#             indsb = torch.cat((indsa, zeros))[1:, :]
#             s_repeat_[~(indsb|indsa)] = 0
#             # indsa corresponds to the first condition of the second equation of the paper
#             s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
#             # indsb corresponds to the second condition of the second equation of the paper
#             s_repeat_[indsb] =  (-s_repeat_ + self.t + self.step)[indsb] / self.step

#             return s_repeat_.sum(1) / size
        
#         classes_size = classes.size()[0]
#         classes_eq = (classes.repeat(classes_size, 1)  == classes.view(-1, 1).repeat(1, classes_size)).data
#         dists = torch.mm(features, features.transpose(0, 1))
#         assert ((dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps).sum().item()) == 0, 'L2 normalization should be used'
#         s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
#         if self.cuda:
#             s_inds= s_inds.cuda()
#         pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
#         neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
#         pos_size = classes_eq[s_inds].sum().item()
#         neg_size = (~classes_eq[s_inds]).sum().item()
#         s = dists[s_inds].view(1, -1)
#         s_repeat = s.repeat(self.tsize, 1)
#         s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()
        
#         histogram_pos = histogram(pos_inds, pos_size)
#         assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1, 
#                             err_msg='Not good positive histogram', verbose=True)
#         histogram_neg = histogram(neg_inds, neg_size)
#         assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1, 
#                             err_msg='Not good negative histogram', verbose=True)
#         histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
#         histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
#         if self.cuda:
#             histogram_pos_inds = histogram_pos_inds.cuda()
#         histogram_pos_repeat[histogram_pos_inds] = 0
#         histogram_pos_cdf = histogram_pos_repeat.sum(0)
#         loss = torch.sum(histogram_neg * histogram_pos_cdf)
        
#         return loss
class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
