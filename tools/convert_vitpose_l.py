"""Chuyển ViTPose-L GỐC (checkpoint tác giả, không phải 'plus') sang định dạng HF VitPoseForPoseEstimation, lưu thành thư mục cục bộ dùng được với build_model.
usage: convert_vitpose_l.py SRC.pth OUT_DIR"""
import sys,os,shutil,torch
from transformers import VitPoseConfig,VitPoseForPoseEstimation
src,out=sys.argv[1:3]
sd=torch.load(src,map_location='cpu',weights_only=True)['state_dict']
c=VitPoseConfig.from_pretrained('usyd-community/vitpose-plus-large');b=c.backbone_config;b.num_experts=1;b.part_features=0   # dùng config plus-large (24 tầng, out_features/stage_names đúng), bỏ MoE
m=VitPoseForPoseEstimation(c);tgt=m.state_dict();new={};H=1024
new['backbone.embeddings.position_embeddings']=sd['backbone.pos_embed']
new['backbone.embeddings.patch_embeddings.projection.weight']=sd['backbone.patch_embed.proj.weight'];new['backbone.embeddings.patch_embeddings.projection.bias']=sd['backbone.patch_embed.proj.bias']
for i in range(24):
    p=f'backbone.blocks.{i}.';q=f'backbone.encoder.layer.{i}.'
    w=sd[p+'attn.qkv.weight'];bb=sd[p+'attn.qkv.bias']
    for j,n in enumerate(('query','key','value')):
        new[q+f'attention.attention.{n}.weight']=w[j*H:(j+1)*H];new[q+f'attention.attention.{n}.bias']=bb[j*H:(j+1)*H]
    new[q+'attention.output.dense.weight']=sd[p+'attn.proj.weight'];new[q+'attention.output.dense.bias']=sd[p+'attn.proj.bias']
    new[q+'layernorm_before.weight']=sd[p+'norm1.weight'];new[q+'layernorm_before.bias']=sd[p+'norm1.bias']
    new[q+'layernorm_after.weight']=sd[p+'norm2.weight'];new[q+'layernorm_after.bias']=sd[p+'norm2.bias']
    for f in ('fc1','fc2'):new[q+f'mlp.{f}.weight']=sd[p+f'mlp.{f}.weight'];new[q+f'mlp.{f}.bias']=sd[p+f'mlp.{f}.bias']
new['backbone.layernorm.weight']=sd['backbone.last_norm.weight'];new['backbone.layernorm.bias']=sd['backbone.last_norm.bias']
hm={'0':'deconv1','3':'deconv2'};bn={'1':'batchnorm1','4':'batchnorm2'}
for k,v in hm.items():new[f'head.{v}.weight']=sd[f'keypoint_head.deconv_layers.{k}.weight']
for k,v in bn.items():
    for s in ('weight','bias','running_mean','running_var','num_batches_tracked'):new[f'head.{v}.{s}']=sd[f'keypoint_head.deconv_layers.{k}.{s}']
new['head.conv.weight']=sd['keypoint_head.final_layer.weight'];new['head.conv.bias']=sd['keypoint_head.final_layer.bias']
miss=[k for k in tgt if k not in new];extra=[k for k in new if k not in tgt];shape=[k for k in new if k in tgt and tuple(new[k].shape)!=tuple(tgt[k].shape)]
print('thiếu',miss[:5],len(miss),'| thừa',extra[:5],len(extra),'| sai shape',shape[:5],len(shape))
assert not miss and not extra and not shape
m.load_state_dict(new);os.makedirs(out,exist_ok=True);m.save_pretrained(out)
for f in ('preprocessor_config.json',):
    from huggingface_hub import hf_hub_download;shutil.copy(hf_hub_download('usyd-community/vitpose-base',f),os.path.join(out,f))
print('đã lưu',out,os.listdir(out))
