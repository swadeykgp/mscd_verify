import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from torchinfo import summary

# Lirpa friendly encoder-decoder
class DoubleConv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__() 
        self.double_conv = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.double_conv(x)
 
class Down(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.conv = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), 
        )
 
    def forward(self, x):
        return self.conv(x)
 
class Up(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__() 
        self.up = nn.Sequential( 
            nn.Upsample(scale_factor=2, mode='nearest'), 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
        ) 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # ensure spatial alignment by cropping or center crop if needed
        if x1.shape[-2:] != x2.shape[-2:]:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
 
class OutConv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)
 
class EncDec_LiRPA(nn.Module): 
    def __init__(self, in_channels=26, out_channels=2): 
        super().__init__() 
        self.inc = DoubleConv(in_channels, 8) 
        self.down1 = Down(8, 16) 
        self.down2 = Down(16, 32) 
        self.down3 = Down(32, 64) 
        self.down4 = Down(64, 128) 
        self.up1 = Up(128, 64) 
        self.up2 = Up(64, 32) 
        self.up3 = Up(32, 16) 
        self.up4 = Up(16, 8) 
        self.outc = OutConv(8, out_channels)
 
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# lirpa friendly Local Attention Encoder-decoder Falconet Architecture   
class AttentionCondenser(nn.Module):
    """Lightweight attention using convolutions instead of full self-attention."""
    def __init__(self, d, reduction=8, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(d, d // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(d // reduction, d, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=dropout)  # Dropout in attention

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.conv2(attn)
        attn = self.dropout(attn)  # Dropout after attention layers
        return x * self.sigmoid(attn)


class ConvAttention(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, reduction_ratio=8):
        super().__init__()
        reduced_dim = hidden_dim // reduction_ratio

        # Downsample with depthwise conv (local feature extraction)
        self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, groups=hidden_dim, bias=False)

        # Pointwise conv (lightweight attention projection)
        self.q_proj = nn.Conv1d(hidden_dim, reduced_dim, 1, bias=False)
        self.k_proj = nn.Conv1d(hidden_dim, reduced_dim, 1, bias=False)
        self.v_proj = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)

        # Attention Scoring
        self.score_proj = nn.Conv1d(reduced_dim, hidden_dim, 1, bias=False)

        # Final pointwise conv
        self.out_proj = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        Output: (batch, seq_len, hidden_dim)
        """
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len) for convs

        # Local feature extraction
        x_dw = self.depthwise_conv(x)

        # Query, Key, Value projection
        q = self.q_proj(x_dw)
        k = self.k_proj(x_dw)
        v = self.v_proj(x_dw)

        # Lightweight attention (sigmoid gating)
        attn_scores = self.score_proj(q * k).sigmoid()

        # Apply attention to values
        attn_output = attn_scores * v

        # Final projection
        out = self.out_proj(attn_output)

        # Residual connection
        out = out + x
        return out.transpose(1, 2)  # Back to (batch, seq_len, hidden_dim)


class MultiHeadConvAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=2, kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # Ensure correct head dim

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.heads = nn.ModuleList([
            ConvAttention(self.head_dim, kernel_size) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        #print(f"embed_dim = {embed_dim}, num_heads * head_dim = {self.num_heads * self.head_dim}")  # Debug

        assert embed_dim == self.num_heads * self.head_dim, \
            f"Expected embedding dim {embed_dim} to match {self.num_heads * self.head_dim} (={self.num_heads * self.head_dim})"

        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)  
        x = x.permute(2, 0, 1, 3)  # (num_heads, batch, seq_len, head_dim)

        x = torch.stack([head(x[i]) for i, head in enumerate(self.heads)], dim=0)

        x = x.permute(1, 2, 0, 3).reshape(batch_size, seq_len, embed_dim)  # Merge heads
        return self.out_proj(x)

 
class FALCONetMHA_LiRPA(nn.Module): 
    def __init__(self, in_channels=26, out_channels=2, dropout=0.1, reduction=8, attention=True, num_heads=4):
        """Init FALCONetMHA_LiRPA fields."""
        super(FALCONetMHA_LiRPA, self).__init__()
        self.dropout = dropout
        self.reduction = reduction
        self.attention = attention
        cur_depth = 8
        self.inc = DoubleConv(in_channels, cur_depth) 
        self.down1 = Down(cur_depth, cur_depth*2) 
        cur_depth *= 2

        self.down2 = Down(cur_depth, cur_depth*2) 
        cur_depth *= 2
        if self.attention:
            self.token_mixer_2 = MultiHeadConvAttention(cur_depth, num_heads=num_heads)
        self.down3 = Down(cur_depth, cur_depth*2) 
        cur_depth *= 2
        if self.attention:
            self.token_mixer_3 = MultiHeadConvAttention(cur_depth, num_heads=num_heads)
        self.down4 = Down(cur_depth, cur_depth*2) 
        cur_depth *= 2
        if self.attention:
            self.token_mixer_4 = MultiHeadConvAttention(cur_depth, num_heads=num_heads)
        self.up1 = Up(cur_depth, int(cur_depth/2)) 
        cur_depth = int(cur_depth/2)
        self.up2 = Up(cur_depth, int(cur_depth/2))
        cur_depth = int(cur_depth/2)
        self.up3 = Up(cur_depth, int(cur_depth/2))
        cur_depth = int(cur_depth/2)
        self.up4 = Up(cur_depth, int(cur_depth/2)) 
        cur_depth = int(cur_depth/2)
        self.outc = OutConv(cur_depth, out_channels)
 
    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.attention:
            # Flatten spatial dimensions
            B, C, H, W = x3.shape
            N = H * W  # Number of tokens
            x3 = x3.view(B, C, N).transpose(1, 2)  # (B, N, C)
            
            # Apply Lightweight Convolutional Multi-Head Attention
            x3 = self.token_mixer_2(x3)  
            
            # Restore spatial shape
            x3 = x3.transpose(1, 2).view(B, C, H, W)  # Restore spatial shape
        x4 = self.down3(x3)
        if self.attention:
            # Flatten spatial dimensions
            B, C, H, W = x4.shape
            N = H * W  # Number of tokens
            x4 = x4.view(B, C, N).transpose(1, 2)  # (B, N, C)
            
            # Apply Lightweight Convolutional Multi-Head Attention
            x4 = self.token_mixer_3(x4)  
            
            # Restore spatial shape
            x4 = x4.transpose(1, 2).view(B, C, H, W)  # Restore spatial shape
        x5 = self.down4(x4)
        if self.attention:
            # Flatten spatial dimensions
            B, C, H, W = x5.shape
            N = H * W  # Number of tokens
            x5 = x5.view(B, C, N).transpose(1, 2)  # (B, N, C)
            
            # Apply Lightweight Convolutional Multi-Head Attention
            x5 = self.token_mixer_4(x5)  
            
            # Restore spatial shape
            x5 = x5.transpose(1, 2).view(B, C, H, W)  # Restore spatial shape
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class AttU_Net(nn.Module):
    def __init__(self,img_ch=26,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sm = nn.Identity()


    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.sm(d1)

       
if __name__ == "__main__": 
    # Example usage
    model, model_name = EncDec_LiRPA(), 'EncDec_LiRPA'
    #x = torch.randn(1, 13, 128, 128) 
    x = torch.randn(1, 26, 256, 256)
    out = model(x)
    print(summary(model, input_size=(x.shape)))
    # Example usage
    model, model_name = FALCONetMHA_LiRPA(2*13, 2 , dropout=0.1, reduction=8, attention=True, num_heads=4), 'FALCONetMHA_LiRPA'
    out = model(x)
    print(summary(model, input_size=(x.shape)))
    
    saved_aunet, saved_aunet_name = AttU_Net(2*13, 2), 'AttU_Net'
    out = saved_aunet(x)
    print(summary(saved_aunet, input_size=(x.shape)))