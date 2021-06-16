import tlt.models
# summary of model flops and parameters

model_list = [tlt.models.lvvit_s,
              tlt.models.lvvit_m,
              tlt.models.lvvit_l]

img_size_list=[224,288,384,448]

for img_size in img_size_list:
    for model_name in model_list:
        model = model_name(img_size=img_size)
        params =  sum([m.numel() for m in model.parameters()])
        flops = model.patch_embed.flops()
        for blk in model.blocks:
            flops = flops + blk.flops(model.patch_embed.num_patches+1)
        print("model: {}, img_size:{},\nparams:{:.2f} M, flops: {:.2f} G \n".format(model_name.__name__, img_size, params/1e6, flops/1e9))

    print('-----------------------')