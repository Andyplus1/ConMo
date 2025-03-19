from math import sqrt
from utilities.utils import isinstance_str
import torch
import torch.nn.functional as F
from einops import rearrange

import torch
import numpy as np
from PIL import Image

# 假设我们有一个布尔 tensor
def bool2img(bool_tensor):  # 生成一个随机的布尔 tensor 作为示例

    # 将布尔 tensor 转换为 uint8 类型图像
    # 将 `True` 转换为 255，并将 `False` 转换为 0
    int_tensor = bool_tensor.cpu().to(torch.uint8) * 255

    # 转换为 NumPy 数组
    int_matrix = int_tensor.numpy()

    # 创建一个 PIL 图像
    mask_image = Image.fromarray(int_matrix, mode='L')

    # 保存图像为 mask.png 或直接展示
    mask_image.save("mask.png")
    # mask_image.show()  # 可选：直接显示图像

def mask_tobboxmask(mask_anchor):
    true_indices_anchor = torch.nonzero(mask_anchor)
    anchor_topleft = true_indices_anchor.min(dim=0).values[1:]
    anchor_bottomright = true_indices_anchor.max(dim=0).values[1:]
    mask_anchor[:,anchor_topleft[0]:anchor_bottomright[0]+1,anchor_topleft[1]:anchor_bottomright[1]+1] = True
    
    return mask_anchor

def mask_or(mask_anchor,mask_all):
    
    # mask_anchor_trangle = mask_tobboxmask(mask_anchor)
    # mask_all_trangle = []
    # for idx in range(mask_all.shape[0]):
    #     mask_this = mask_all[idx]
    #     mask_this_trangle = mask_tobboxmask(mask_this)
    #     mask_all_trangle.append(mask_this_trangle)
    # mask_all_trangle = torch.stack(mask_all_trangle)
    # 可视化mask_anchor_trangle
    # bool2img(mask_anchor_trangle)
   

    return mask_all | mask_anchor

def mask_or_interfere(mask_anchor,mask_all,i,j,mask_other):
    
    # mask_anchor_trangle = mask_tobboxmask(mask_anchor)
    # mask_all_trangle = []
    mask_final = []
    mask_other_current = mask_other[j][i]

    for idx in range(mask_all.shape[0]):
        mask_other_this = mask_other[j][idx]
        mask_all_this = mask_all[idx]
        mask_part1 = mask_anchor^(mask_anchor&mask_other_this)
        mask_part2 = mask_all_this^(mask_all_this&mask_other_current)
        mask_final.append(mask_part1|mask_part2)
        # mask_final.append((mask_anchor&mask_other_this)|(mask_all_this&mask_other_current))
    
    mask_final = torch.stack(mask_final)
    return mask_final


    # for idx in range(mask_all.shape[0]):
    #     mask_this = mask_all[idx]
    #     mask_this_trangle = mask_tobboxmask(mask_this)
    #     mask_all_trangle.append(mask_this_trangle)
    # mask_all_trangle = torch.stack(mask_all_trangle)
    # 可视化mask_anchor_trangle
    # bool2img(mask_anchor_trangle)
   

    # return mask_all | mask_anchor

def mask_or_bbox(mask_anchor,mask_all):
    
    mask_anchor_trangle = mask_tobboxmask(mask_anchor)
    mask_all_trangle = []
    for idx in range(mask_all.shape[0]):
        mask_this = mask_all[idx]
        mask_this_trangle = mask_tobboxmask(mask_this)
        mask_all_trangle.append(mask_this_trangle)
    mask_all_trangle = torch.stack(mask_all_trangle)
    # 可视化mask_anchor_trangle
    # bool2img(mask_anchor_trangle)
   

    return mask_all_trangle | mask_anchor_trangle

def new_mean(tensor,mask):
    f,c,h,w = tensor.shape
    expanded_mask = mask.expand(-1, c, -1, -1)
    selected_values = tensor*expanded_mask
    
    sum_values = selected_values.sum(dim=(2,3))
    true_counts = expanded_mask.view(f, c, -1).sum(dim=2) 
    
    mean_values = sum_values / true_counts
    mean_values = mean_values.unsqueeze(-1).unsqueeze(-1)
    return mean_values

def scale_bbox(top_left: tuple, bottom_right: tuple, scale_x: float, scale_y: float) -> tuple:
    """
    根据给定的比例缩放边界框
    
    参数:
        top_left (tuple): 左上点的坐标 (x1, y1)
        bottom_right (tuple): 右下点的坐标 (x2, y2)
        scale_x (float): x 方向的缩放比例
        scale_y (float): y 方向的缩放比例
        
    返回:
        tuple: 新的边界框坐标 (new_top_left, new_bottom_right)
    """
    # 计算中心点
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    
    # 计算半宽和半高
    half_width = (bottom_right[0] - top_left[0]) / 2
    half_height = (bottom_right[1] - top_left[1]) / 2
    
    # 根据缩放比例计算新的半宽和半高
    new_half_width = half_width * scale_x
    new_half_height = half_height * scale_y
    
    # 计算新的左上角和右下角坐标
    new_top_left = (center_x - new_half_width, center_y - new_half_height)
    new_bottom_right = (center_x + new_half_width, center_y + new_half_height)
    
    return new_top_left, new_bottom_right

def expand_true_regions(matrix,h_raitio,w_ratio): 
    if not isinstance(matrix, torch.Tensor) or matrix.dim() != 4:
        raise ValueError("输入必须是形状为 (n, c, h, w) 的四维 PyTorch Tensor")

    # 创建一个与原矩阵相同形状的全 False 矩阵
    expanded_matrix = torch.zeros_like(matrix)

    # 找到所有为 True 的索引
    # true_indices = torch.nonzero(matrix)
    

    for j in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            
            # from IPython import embed;embed()
            matrix_this = matrix[j][i]
            true_indices_this = torch.nonzero(matrix_this)
            # print("test_"+str(i)+'_'+str(j))
            # print(true_indices_this)
            if len(true_indices_this) == 0:
                continue
            h_min = true_indices_this[:,0].min()
            h_max = true_indices_this[:,0].max()
            w_min = true_indices_this[:,1].min()
            w_max = true_indices_this[:,1].max()
            (x1,y1),(x2,y2) = scale_bbox((w_min,h_min),(w_max,h_max),w_ratio,h_raitio)
            x1 = max(0,int(x1))
            y1 = max(0,int(y1))
            x2 = min(matrix.shape[-1]-1,int(x2))
            y2 = min(matrix.shape[-2]-1,int(y2))
            expanded_matrix[j,i,y1:y2+1,x1:x2+1] = True

        
        # 找到扩展后每个 true 区域的范围


    return expanded_matrix
    
@torch.autocast(device_type="cuda", dtype=torch.float32)
def calculate_losses(orig_features, target_features, masks,masks_target,config):
    orig = orig_features
    target = target_features

    orig = orig.detach()

    total_loss = 0
    losses = {}
    
    if config["features_loss_weight"] > 0:
        if config["global_averaging"]:
            orig = orig.mean(dim=(2, 3), keepdim=True)
            target = target.mean(dim=(2, 3), keepdim=True)

        features_loss = compute_feature_loss(orig, target)
        total_loss += config["features_loss_weight"] * features_loss
        losses["features_mse_loss"] = features_loss

    if config["features_diff_loss_weight"] > 0: #True
        
        features_diff_loss = 0
        
        # orig = orig[:,:,:5,:]
        # target = torch.cat((target[:,:,6:,:],target[:,:,:7,:]),dim=2)

        # change begin
        
        # orig1 = orig[:,:,:4,:]
        # orig2 = orig[:,:,4:,:]
        
        # bool_masks = masks>0.5
        if config["resize_way"] == "nearest":
            compressed_tensor = F.interpolate(masks.float().squeeze(2), size=orig.shape[-2:], mode='nearest')
            mask_bool_4gen =  F.interpolate(masks_target.float().squeeze(2), size=orig.shape[-2:],mode='nearest')
            
        elif config["resize_way"] == "bilinear":
            compressed_tensor = F.interpolate(masks.float().squeeze(2), size=orig.shape[-2:], mode='bilinear')
            mask_bool_4gen =  F.interpolate(masks_target.float().squeeze(2), size=orig.shape[-2:],mode='bilinear')
            
            
        mask_bool = compressed_tensor > 0.5
        mask_bool_4gen =  mask_bool_4gen>0.5
        
        # mask_bool_4gen =  mask_bool
        # mask_bool_4gen = expand_true_regions(mask_bool,config.h_rescale,config.w_rescale)
        
        mask_other = mask_bool.transpose(0,1)
        mask_other_list = []
        for i in range(mask_bool.shape[0]):
            mask_other_this = torch.cat((mask_other[:,:i],mask_other[:,i+1:]),dim=1)
            mask_other_this = mask_other_this.any(dim=1)
            mask_other_list.append(mask_other_this)  

        
        mask_other = torch.stack(mask_other_list)
        loss_weight = 1
        if config['reposition']:
            loss_weight+=1

        
         
        # mask有黑的时考虑下怎么处理
        for i in range(mask_bool.shape[1]):
            mask_target_all = torch.zeros_like(mask_bool[0])
            mask_target_all_4gen= torch.zeros_like(mask_bool[0])
            for j in range(mask_bool.shape[0]):
                
                mask_anchor1 = mask_bool[j][i]  #torch.Size([2, 24, 10, 18])
                mask_anchor1_4gen = mask_bool_4gen[j][i]

                if config["bbox_mask"]:
                    mask_target = mask_or_bbox(mask_anchor1,mask_bool[j])
                    mask_target_4gen = mask_or_bbox(mask_anchor1_4gen,mask_bool_4gen[j])
                    mask_target_all = mask_target_all | mask_target
                    mask_target_all_4gen = mask_target_all_4gen | mask_target_4gen
                elif config["interfere_mask"]:
                    mask_target = mask_or_interfere(mask_anchor1,mask_bool[j],i,j,mask_other)
                    mask_target_4gen = mask_or_interfere(mask_anchor1_4gen,mask_bool_4gen[j],i,j,mask_other)
                    mask_target_bask = mask_or(mask_anchor1,mask_bool[j])
                    mask_target_4gen_bask = mask_or(mask_anchor1_4gen,mask_bool_4gen[j])
                    mask_target_all = mask_target_all | mask_target_bask
                    mask_target_all_4gen = mask_target_all_4gen | mask_target_4gen_bask
                else:
                    mask_target = mask_or(mask_anchor1,mask_bool[j])
                    mask_target_4gen = mask_or(mask_anchor1_4gen,mask_bool_4gen[j])
                    mask_target_all = mask_target_all | mask_target
                    mask_target_all_4gen = mask_target_all_4gen | mask_target_4gen
                # print("test2")
            orig_anchor1 = orig[i].squeeze(0).repeat(mask_bool.shape[1],1,1,1)
            target_anchor1 = target[i].squeeze(0).repeat(mask_bool.shape[1],1,1,1)

            orig_unmasked = new_mean(orig, ~mask_target_all.unsqueeze(1))
            orig_anchor1_unmasked = new_mean(orig_anchor1, ~mask_target_all.unsqueeze(1))
            orig_diffs_unmasked = orig_unmasked - orig_anchor1_unmasked
            target_unmasked = new_mean(target, ~mask_target_all_4gen.unsqueeze(1))
            target_anchor1_unmasked = new_mean(target_anchor1, ~mask_target_all_4gen.unsqueeze(1))
            target_diffs_unmasked = target_unmasked - target_anchor1_unmasked
            # features_diff_loss += 1 - F.cosine_similarity(target_diffs_unmasked, orig_diffs_unmasked, dim=1).mean()

            # 

            mask_target_all = torch.zeros_like(mask_bool[0])
            mask_target_all_4gen= torch.zeros_like(mask_bool[0])

            mask_back = torch.ones_like(mask_bool[0])
            # from IPython import embed;embed()

            for j in range(mask_bool.shape[0]):
                
                mask_anchor1 = mask_bool[j][i]  #torch.Size([2, 24, 10, 18])
                mask_anchor1_4gen = mask_bool_4gen[j][i]

                if config["bbox_mask"]:
                    mask_target = mask_or_bbox(mask_anchor1,mask_bool[j])
                    mask_target_4gen = mask_or_bbox(mask_anchor1_4gen,mask_bool_4gen[j])
                    mask_target_all = mask_target_all | mask_target
                    mask_target_all_4gen = mask_target_all_4gen | mask_target_4gen
                elif config["interfere_mask"]:
                    mask_target = mask_or_interfere(mask_anchor1,mask_bool[j],i,j,mask_other)
                    mask_target_4gen = mask_or_interfere(mask_anchor1_4gen,mask_bool_4gen[j],i,j,mask_other)
                    mask_target_bask = mask_or(mask_anchor1,mask_bool[j])
                    mask_target_4gen_bask = mask_or(mask_anchor1_4gen,mask_bool_4gen[j])
                    mask_target_all = mask_target_all | mask_target_bask
                    mask_target_all_4gen = mask_target_all_4gen | mask_target_4gen_bask
                else:
                    mask_target = mask_or(mask_anchor1,mask_bool[j])
                    mask_target_4gen = mask_or(mask_anchor1_4gen,mask_bool_4gen[j])
                    mask_target_all = mask_target_all | mask_target
                    mask_target_all_4gen = mask_target_all_4gen | mask_target_4gen
                 
                orig_anchor1 = orig[i].squeeze(0).repeat(mask_bool.shape[1],1,1,1)
                target_anchor1 = target[i].squeeze(0).repeat(mask_bool.shape[1],1,1,1)

                orig_masked = (new_mean(orig, mask_target.unsqueeze(1))+config["cm_weight"][j]*orig_unmasked)/(1+config["cm_weight"][j])
                orig_anchor1_masked = (new_mean(orig_anchor1, mask_target.unsqueeze(1))+config["cm_weight"][j]*orig_anchor1_unmasked)/(1+config["cm_weight"][j])
                orig_diffs_masked = orig_masked - orig_anchor1_masked
                target_masked = new_mean(target, mask_target_4gen.unsqueeze(1))
                target_anchor1_masked = new_mean(target_anchor1, mask_target_4gen.unsqueeze(1))
                target_diffs_masked = target_masked - target_anchor1_masked

                if config['reposition']:
                    mask_target_4erase = mask_target^(mask_target&mask_target_4gen)
                    target_erased = new_mean(target, mask_target_4erase.unsqueeze(1))
                    target_anchor1_erased = new_mean(target_anchor1, mask_target_4erase.unsqueeze(1))
                    target_diffs_erased = target_masked - target_anchor1_masked
                    # features_diff_loss += (1 - F.cosine_similarity(target_diffs_erased, orig_diffs_unmasked, dim=1).mean())*2

                
                
                # if config['erase'][j]:
                #     features_diff_loss += 1 - F.cosine_similarity(target_diffs_masked, orig_diffs_unmasked, dim=1).mean()
                # else:
                #     features_diff_loss += 1 - F.cosine_similarity(target_diffs_masked, orig_diffs_masked, dim=1).mean()
                # target_back = new_mean(target, mask_back.unsqueeze(1))
                # target_anchor1_back  = new_mean(target_anchor1,mask_back.unsqueeze(1))
                # target_diffs_back  = target_back - target_anchor1_back 
                # print("erase")
                # from IPython import embed;embed()
                features_diff_loss += 1 - F.cosine_similarity(target_diffs_masked, orig_diffs_masked, dim=1).mean()
            features_diff_loss += 1 - F.cosine_similarity(target_diffs_unmasked, orig_diffs_unmasked, dim=1).mean()
                

                # features_diff_loss += 1 - F.cosine_similarity(target_diffs_masked, orig_diffs_unmasked, dim=1).mean()

        # features_diff_loss /= (len(orig)*(1+(mask_bool.shape[0])*loss_weight))
        features_diff_loss /= (len(orig)*(1+(mask_bool.shape[0])))

        total_loss += config["features_diff_loss_weight"] * features_diff_loss
        losses["features_diff_loss"] = features_diff_loss

    losses["total_loss"] = total_loss
    return losses


def compute_feature_loss(orig, target):
    features_loss = 0
    for i, (orig_frame, target_frame) in enumerate(zip(orig, target)):
        features_loss += 1 - F.cosine_similarity(target_frame, orig_frame.detach(), dim=0).mean()
    features_loss /= len(orig)
    return features_loss


def get_timesteps(scheduler, num_inference_steps, max_guidance_timestep, min_guidance_timestep):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * max_guidance_timestep), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = int(num_inference_steps * min_guidance_timestep)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    if t_end > 0:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order : -t_end * scheduler.order]
    else:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order :]
    return timesteps, guidance_schedule


def register_time(model, t):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "t", t)


def register_batch(model, b):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "b", b)


def register_guidance(model):
    guidance_schedule = model.guidance_schedule
    num_frames = model.video.shape[0]
    h = model.video.shape[2]
    w = model.video.shape[3]

    class ModuleWithConvGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "spatial_convolution",
            ]
            self.module_type = module_type
            if self.module_type == "spatial_convolution":
                self.starting_shape = "(b t) d h w"
            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config
            self.saved_features = None

        def forward(self, input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.module.norm1(hidden_states)
            hidden_states = self.module.nonlinearity(hidden_states)

            if self.module.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.module.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.module.downsample is not None:
                input_tensor = self.module.downsample(input_tensor)
                hidden_states = self.module.downsample(hidden_states)

            hidden_states = self.module.conv1(hidden_states)

            if temb is not None:
                temb = self.module.time_emb_proj(self.module.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.module.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.module.norm2(hidden_states)

            if temb is not None and self.module.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.module.nonlinearity(hidden_states)

            hidden_states = self.module.dropout(hidden_states)
            hidden_states = self.module.conv2(hidden_states)

            if self.config["guidance_before_res"] and (self.t in self.guidance_schedule):
                self.saved_features = rearrange(
                    hidden_states, f"{self.starting_shape} -> b t d h w", t=self.num_frames
                )

            if self.module.conv_shortcut is not None:
                input_tensor = self.module.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.module.output_scale_factor

            if not self.config["guidance_before_res"] and (self.t in self.guidance_schedule):
                self.saved_features = rearrange(
                    output_tensor, f"{self.starting_shape} -> b t d h w", t=self.num_frames
                )

            return output_tensor

    class ModuleWithGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "temporal_attention",
                "spatial_attention",
                "temporal_convolution",
                "upsampler",
            ]
            self.module_type = module_type
            if self.module_type == "temporal_attention":
                self.starting_shape = "(b h w) t d"
            elif self.module_type == "spatial_attention":
                self.starting_shape = "(b t) (h w) d"
            elif self.module_type == "temporal_convolution":
                self.starting_shape = "(b t) d h w"
            elif self.module_type == "upsampler":
                self.starting_shape = "(b t) d h w"
            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config

        def forward(self, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = self.module(x, *args, **kwargs)
            t = self.num_frames
            if self.module_type == "temporal_attention":
                size = out.shape[0] // self.b
            elif self.module_type == "spatial_attention":
                size = out.shape[1]
            elif self.module_type == "temporal_convolution":
                size = out.shape[2] * out.shape[3]
            elif self.module_type == "upsampler":
                size = out.shape[2] * out.shape[3]

            if self.t in self.guidance_schedule:
                h, w = int(sqrt(size * self.h / self.w)), int(sqrt(size * self.h / self.w) * self.w / self.h)
                self.saved_features = rearrange(
                    out, f"{self.starting_shape} -> b t d h w", t=self.num_frames, h=h, w=w
                )

            return out

    up_res_dict = model.config["up_res_dict"]
    for res in up_res_dict:
        module = model.unet.up_blocks[res]
        samplers = module.upsamplers
        if model.config["use_upsampler_features"]:
            if samplers is not None:
                for i in range(len(samplers)):
                    submodule = samplers[i]
                    samplers[i] = ModuleWithGuidance(
                        submodule,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=f"decoder_res{res}_upsampler",
                        config=model.config,
                        module_type="upsampler",
                    )
        for block in up_res_dict[res]:
            block_name = f"decoder_res{res}_block{block}"
            if model.config["use_conv_features"]:
                block_name_conv = f"{block_name}_spatial_convolution"
                submodule = module.resnets[block]
                module.resnets[block] = ModuleWithConvGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=model.config,
                    module_type="spatial_convolution",
                )

            if model.config["use_temp_conv_features"]:
                block_name_conv = f"{block_name}_temporal_convolution"
                submodule = module.temp_convs[block]
                module.temp_convs[block] = ModuleWithGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=model.config,
                    module_type="temporal_convolution",
                )

            if res == 0:  # UpBlock3D does not have attention
                if model.config["use_spatial_attention_features"]:
                    block_name_spatial = f"{block_name}_spatial_attn1"
                    submodule = module.attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=block_name_spatial,
                        config=model.config,
                        module_type="spatial_attention",
                    )
                if model.config["use_temporal_attention_features"]:
                    submodule = module.temp_attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    block_name_temp = f"{block_name}_temporal_attn1"
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=model.config,
                        module_type="temporal_attention",
                    )
                    block_name_temp = f"{block_name}_temporal_attn2"
                    submodule.attn2 = ModuleWithGuidance(
                        submodule.attn2,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=model.config,
                        module_type="temporal_attention",
                    )
