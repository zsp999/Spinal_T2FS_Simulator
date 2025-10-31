import torch
import torch.nn as nn
import torch.nn.functional as F


# 
def up(x): 
    return nn.functional.interpolate(x,scale_factor=2)
        
def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def conv_decod_block(in_dim, out_dim, act_fn):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )


def _upsample_like(src,tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear',align_corners=True)
    
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, activation):
        super(Self_Attn,self).__init__()

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):

        m_batchsize,C,width,height = x.size()
        proj_query  = self.query_conv(x).reshape(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).reshape(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check 
        proj_value = self.value_conv(x).reshape(m_batchsize,-1,width*height) # B X C X N
        out =  self.gamma*torch.bmm(proj_value, F.softmax(energy,dim=1).permute(0,2,1)).reshape(m_batchsize,C,width,height)+ x
        return out



class MixedFusion_Block(nn.Module):
    
    def __init__(self,in_dim, out_dim,act_fn):
        super(MixedFusion_Block, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),act_fn)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)

    def forward(self, x1, x2,xx):
        
        fusion_sum = x1+x2
        fusion_mul = x1*x2
        fusion_max = torch.cat((x1.unsqueeze(dim=1), x2.unsqueeze(dim=1)),dim=1).max(dim=1)[0]
         
        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max),dim=1)
        out1 = self.layer1(out_fusion)
        out2 = self.layer2(torch.cat((out1, xx),dim=1))
        
        return out2
        

class MixedFusion_Block0(nn.Module):
    def __init__(self,in_dim, out_dim,act_fn):
        super(MixedFusion_Block0, self).__init__()
        
        self.layers = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),act_fn,
                                    nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn)


    def forward(self, x1,x2): 

        fusion_sum = x1+x2
        fusion_mul = x1*x2
        fusion_max = torch.cat((x1.unsqueeze(dim=1), x2.unsqueeze(dim=1)),dim=1).max(dim=1)[0]
        out_fusion = torch.cat((fusion_sum,fusion_mul,fusion_max),dim=1)
        return self.layers(out_fusion)

class Down_block(nn.Module):
    def __init__(self,in_dim, out_dim, act_fn):
        super(Down_block, self).__init__()
        
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_dim,  out_channels=out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim), act_fn,
                nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim), act_fn)

    def forward(self, x): 
        return self.layers(x)


class Multi_modal_generator(nn.Module):

    def __init__(self):
        super(Multi_modal_generator,self).__init__()
        
        # 1, 1, 64
        self.in_dim = 1
        self.out_dim = 64
        self.final_out_dim = 1
        act_fn = nn.LeakyReLU(inplace=True) #0.2, 
        act_fn2 = nn.ReLU(inplace=True)
        

        self.down_1_0 = Down_block(self.in_dim, self.out_dim, act_fn)
        self.down_1_1 = Down_block(self.in_dim, self.out_dim, act_fn)
        self.down_2_0 = Down_block(self.out_dim*3, self.out_dim*2, act_fn) 
        self.down_2_1 = Down_block(self.out_dim*3, self.out_dim*2, act_fn)
        self.down_3_0 = Down_block(self.out_dim*8, self.out_dim*4, act_fn)
        self.down_3_1 = Down_block(self.out_dim*8, self.out_dim*4, act_fn)

        self.down_fu_1 = MixedFusion_Block0(self.out_dim,self.out_dim*2,act_fn)
        self.down_fu_2 = MixedFusion_Block(self.out_dim*2,self.out_dim*4,act_fn)
        self.down_fu_3 = MixedFusion_Block(self.out_dim*4,self.out_dim*4,act_fn)
        self.down_fu_4 = nn.Sequential(nn.Conv2d(in_channels=self.out_dim*4,  out_channels=self.out_dim*8, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.out_dim*8), act_fn,)     
        self.deconv_1_0 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn2)
        self.deconv_2_0 = MixedFusion_Block(self.out_dim * 4, self.out_dim * 2,act_fn2)
        # self.attn_deconv_2_0 = Self_Attn(self.out_dim*2, act_fn2) #去掉

        self.deconv_3_0 = MixedFusion_Block(self.out_dim * 2, self.out_dim * 1,act_fn2)
        self.deconv_4_0 = MixedFusion_Block(self.out_dim * 1, self.out_dim,act_fn2)  
        self.deconv_5_0 = conv_decod_block(self.out_dim * 1, int(self.out_dim*1),act_fn2) 
        self.out = nn.Sequential(nn.Conv2d(int(self.out_dim * 9),1, kernel_size=3, stride=1, padding='same'),nn.Tanh()) # 

                
    def forward(self, inputs):


        down_1_0 = self.down_1_0(inputs[:,0:1,:,:]) #（1,32,256,224）
        down_1_1 = self.down_1_1(inputs[:,1:2,:,:]) #（1,32,256,224）
                                                                                    
        down_fu_1m  = maxpool()(self.down_fu_1(down_1_0, down_1_1)) #（1,64,128,112） 
        down_fu_1m_2  = maxpool()(down_fu_1m) #(1,64,64,56) 

        down_2_0 = self.down_2_0(torch.cat((maxpool()(down_1_0), down_fu_1m),dim=1))  #（1,64,128,112）  
        down_2_1 = self.down_2_1(torch.cat((maxpool()(down_1_1), down_fu_1m),dim=1))  #（1,64,128,112）  
        down_2_0m = maxpool()(down_2_0)  #（1,64,64,56）
        down_2_1m = maxpool()(down_2_1)  #（1,64,64,56）

                                                                                 
        down_fu_2m  = maxpool()(self.down_fu_2(down_2_0, down_2_1,  down_fu_1m))  #（1,128,64,56） 
        down_3_0 = self.down_3_0(torch.cat((down_2_0m, down_fu_2m, down_fu_1m_2),dim=1))  #（1,128,64,56） 
        down_3_1 = self.down_3_1(torch.cat((down_2_1m, down_fu_2m, down_fu_1m_2),dim=1))  #（1,128,64,56） 
        down_fu_4   = self.down_fu_4(self.down_fu_3(down_3_0, down_3_1,  down_fu_2m)) #（1,256,64,56） 

        deconv_1_0 = self.deconv_1_0(down_fu_4) #（1,128,64,56） 
        deconv_2_0 = self.deconv_2_0(down_3_0, down_3_1, deconv_1_0)
        # deconv_2_0 = self.attn_deconv_2_0(self.deconv_2_0(down_3_0, down_3_1, deconv_1_0)) #（1,64,64,56） 
        deconv_3_0 = self.deconv_3_0(down_2_0, down_2_1, up(deconv_2_0))# #（1,32,128,112）
        deconv_4_0 = self.deconv_4_0(down_1_0, down_1_1, up(deconv_3_0)) #（1,32,256,224）   
        deconv_5_0 = self.deconv_5_0(deconv_4_0) #（1,32,256,224）  
        output  = self.out(torch.cat((deconv_5_0, _upsample_like(deconv_1_0, deconv_5_0), _upsample_like(deconv_2_0, deconv_5_0), \
                    _upsample_like(deconv_3_0, deconv_5_0), deconv_4_0), dim=1))
        return output
 
 

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)]
            layers.append(nn.BatchNorm2d(out_features))  #,0.8
            layers.append(nn.LeakyReLU( inplace=True)) #0.2,

            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 32),#changed 256-128
            *discrimintor_block(32, 64), #128-64
            *discrimintor_block(64, 128), #64-32
            *discrimintor_block(128, 256), #32-16
            nn.Conv2d(256, 1, kernel_size=3), 
            
        )

    def forward(self, img):
        return self.model(img)    


class MyLambdaLR():
    def __init__(self, n_epochs, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0)
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

