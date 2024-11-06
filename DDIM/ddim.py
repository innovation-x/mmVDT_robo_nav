import torch
import torch.nn as nn
from TimeSformer.VDT_vm import VDT
from TimeSformer.VDT_cat_vm import VDT_cat

from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # 去掉 'module.' 前缀
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

class ddim_generator(nn.Module):
    def __init__(self,nit=200, nhead=8,patch_size=8, depth=12, hidden_size=768, input_size=64,in_channels=9, sequence_length=16, model_path=None, _cuda=None, cat=False):
        super(ddim_generator, self).__init__()

        self._cuda = _cuda
        self.nit = nit 
        self.beta = torch.linspace(1e-4, 0.02, self.nit) 
        #beta = _beta[::1000//nit]
        self.alpha = 1.0-self.beta
        self.alpha_cum = torch.tensor([self.alpha[0]] + [torch.prod(self.alpha[:i+1]) for i in range(1, len(self.beta))])

        self.trans_model = VDT(
                input_size=input_size,
                patch_size=patch_size,
                in_channels=in_channels,
                hidden_size=hidden_size,
                depth=depth,
                num_frames=sequence_length,
                num_heads=nhead,
                mlp_ratio=4.0,
                learn_sigma=False
            )
        
        if cat:
            self.trans_model = VDT_cat(
                input_size=input_size,
                patch_size=patch_size,
                in_channels=in_channels,
                hidden_size=hidden_size,
                depth=depth,
                num_frames=sequence_length,
                num_heads=nhead,
                mlp_ratio=4.0,
                learn_sigma=False
            )
        
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
            has_module_prefix = any(key.startswith('module.') for key in checkpoint.keys())
            if has_module_prefix:
                checkpoint = remove_module_prefix(checkpoint)
            self.trans_model.load_state_dict(checkpoint)
            print ("load model: success")
        if _cuda is not None:
            self.trans_model.cuda(_cuda)
        self._cuda = _cuda
        #if not train:
        self.trans_model.eval()

    def forward(self, noised_vid, noised_mot,mask_vid,mask_motion, ncal=10, mask_out=True):
        # vid (1,16,64,64,9) mot (1,16,2)
        time = torch.tensor([0]* noised_vid.size(0)).view(-1,1).cuda(self._cuda)
        known_vid = noised_vid.clone()
        known_mot = noised_mot.clone()
        known_vid[mask_vid.byte()]=0
        known_mot[mask_motion.byte()]=0

        cand = list(range(self.nit))[::self.nit//ncal][::-1]
        with torch.no_grad():
            for idx, _d in enumerate(cand):
                time[:,0] = _d
                if _d>0:
                    _nd = cand[idx+1]
                    noised_vid = noised_vid.permute(0,4,1,2,3)
                    # print(noised_vid.shape, noised_mot.shape)
                    # print(noised_vid.shape, noised_mot.shape, time.shape)
                    time = time.squeeze(-1)
                    et_v, et_m = self.trans_model(noised_vid,noised_mot,time)
                    time = time.unsqueeze(-1)
                    at = self.alpha_cum[_d]
                    at_next = self.alpha_cum[_nd]
                    noised_vid0_t = (noised_vid - et_v * torch.sqrt(1-at))/torch.sqrt(at)
                    noised_mot0_t = (noised_mot - et_m * torch.sqrt(1-at))/torch.sqrt(at)
                    # c1 = torch.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
                    c2 = torch.sqrt((1 - at_next))
                    noised_vid = torch.sqrt(at_next) * noised_vid0_t + c2 * et_v
                    noised_mot = torch.sqrt(at_next) * noised_mot0_t + c2 * et_m
                    
                else:
                    noised_vid = noised_vid.permute(0,4,1,2,3)
                    if ncal == 1:
                        time = torch.tensor([self.nit-1]).reshape(noised_vid.size(0),1).cuda(self._cuda)
                        _d = self.nit -1
                    
                    time = time.squeeze(-1)
                    et_v, et_m = self.trans_model(noised_vid,noised_mot,time)
                    at = self.alpha_cum[_d]
                    noised_vid = (noised_vid - et_v * torch.sqrt(1-at))/torch.sqrt(at)
                    noised_mot =(noised_mot - et_m * torch.sqrt(1-at))/torch.sqrt(at)

                noised_vid = noised_vid.permute(0,2,3,4,1)
                noised_vid[1-mask_vid.byte()] = 0
                noised_vid += known_vid

                noised_mot[1-mask_motion.byte()] = 0
                noised_mot += known_mot
        
        return noised_vid, noised_mot



if __name__ == '__main__':
    modelData = '/home/xu/xu/Diffusion_nav_robo/checkpoints/DiffT_80.pth.tar'
    ddim_generator = ddim_generator(mask_start=8, mask_end=16, model_path=modelData, _cuda=0)
    batch = torch.randn(1,16,64,64,9).cuda(0)
    mot = torch.randn(1,2,16).permute(0,2,1).cuda(0)
    a,b = ddim_generator(batch, mot)