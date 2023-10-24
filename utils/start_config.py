import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

def print_options(args, model):
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.FL_platform, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '


    ## save to disk of current log
    args.file_name = os.path.join(args.output_dir, 'log_file.txt')

    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)


def initization_configure(args, vis= False):

    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = args.gpu_ids) if torch.cuda.is_available() else "cpu")

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.device == 'cpu':
        torch.cuda.manual_seed(args.seed)

    # select class number 
    dataset_class_map = {
        'cifar10': 10,
        'pacs': 7,
        'gldk23': 203,
        'isic19': 8
    }
    args.num_classes = dataset_class_map.get(args.dataset, 2)


    # select and initialize model 
    if "ResNet" in args.FL_platform:

        if 'LN' in args.norm: 
            print('Architecture: ResNet-50 with LN')
            import torchvision.models as torch_models
            model = torch_models.resnet50(norm_layer=nn.LayerNorm)
            # load BN weights from checkpoint
            checkpoint = 'additional_weights/resnet50-0676ba61.pth'
            # incompatible keys will be discarded
            model.load_state_dict(torch.load(checkpoint), strict=False)

        elif 'GN' in args.norm: 
            print('Architecture: ResNet-50 with GN')
            from timm.models import resnet50_gn
            model = resnet50_gn(pretrained=args.pretrained) 

        else:
            print('Architecture: ResNet-50')
            from timm.models import resnet50
            model = resnet50(pretrained=args.pretrained)

        model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "EfficientNet" in args.FL_platform:
        print('Architecture: EfficientNet-B5')
        from timm.models import efficientnet_b5
        model = efficientnet_b5(pretrained=args.pretrained)
        model.classifier = nn.Linear(model.classifier.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "ConvNeXt" in args.FL_platform:
        print('Architecture: ConvNeXt-tiny')
        from timm.models import convnext_tiny
        model = convnext_tiny(pretrained=args.pretrained)
        model.head.fc = nn.Linear(model.head.fc.weight.shape[0], args.num_classes)
        model.to(args.device)

    elif "MaxViT" in args.FL_platform:
        print('Architecture: MaxViT-tiny')
        from timm.models.maxxvit import maxvit_tiny_rw_224
        model = maxvit_tiny_rw_224(pretrained=args.pretrained)
        model.head.fc = nn.Linear(model.head.fc.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "ViT" in args.FL_platform:
        print('Architecture: ViT-small')
        from timm.models.vision_transformer import vit_small_patch16_224
        model = vit_small_patch16_224(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "DeiT" in args.FL_platform:
        print('Architecture: DeiT-small')
        from timm.models.deit import deit_small_patch16_224
        model = deit_small_patch16_224(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "Swin-V1" in args.FL_platform:
        print('Architecture: Swin-V1-tiny')
        from timm.models.swin_transformer import swin_tiny_patch4_window7_224
        model = swin_tiny_patch4_window7_224(pretrained=args.pretrained)
        model.head.fc = nn.Linear(model.head.fc.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "Swin-V2" in args.FL_platform:
        print('Architecture: Swin-V2-tiny')
        from timm.models.swin_transformer_v2_cr import swinv2_cr_tiny_ns_224
        model = swinv2_cr_tiny_ns_224(pretrained=args.pretrained)
        model.head.fc = nn.Linear(model.head.fc.weight.shape[0], args.num_classes)
        model.to(args.device)
        
    elif "ConvMixer" in args.FL_platform:
        print('Architecture: ConvMixer-768_32')
        from timm.models.convmixer import convmixer_768_32
        model = convmixer_768_32(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    elif "CAFormer" in args.FL_platform:
        print('Architecture: CAFormer-S18')
        from metaformer.metaformer_baselines import caformer_s18
        model = caformer_s18(pretrained=args.pretrained)
        model.head.fc2 = nn.Linear(model.head.fc2.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    elif "ConvFormer" in args.FL_platform:
        print('Architecture: ConvFormer-S18')
        from metaformer.metaformer_baselines import convformer_s18
        model = convformer_s18(pretrained=args.pretrained)
        model.head.fc2 = nn.Linear(model.head.fc2.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif 'PoolFormer' in args.FL_platform:
            
        if 'LN' in args.norm:
            print('Architecture: Poolformer-S12 with LN')
            from poolformer.models.poolformer import poolformer_s12
            from poolformer.models.poolformer import LayerNormChannel
            model = poolformer_s12(norm_layer=LayerNormChannel)
            check = torch.load('additional_weights/poolformer_ln_s12.pth.tar')
            model.load_state_dict(check)
            
        elif 'BN' in args.norm:
            print('Architecture: Poolformer-S12 with BN')
            from poolformer.models.poolformer import poolformer_s12
            model = poolformer_s12(norm_layer=torch.nn.BatchNorm2d)
            check = torch.load('additional_weights/poolformer_bn_s12.pth.tar')
            model.load_state_dict(check)

        elif 'GN' in args.norm:
            print('Architecture: Poolformer-S12 with GN')
            from poolformer.models.poolformer import poolformer_s12
            from utils.architectures_modifications import poolformer_to_group_norm
            model = poolformer_s12()
            check = torch.load('additional_weights/poolformer_s12.pth.tar')
            model.load_state_dict(check, strict=True)
            poolformer_to_group_norm(model)
            print(model)
        
        else:
            print('Architecture: PoolFormer-S36')
            from timm.models.metaformer import poolformer_s36
            model = poolformer_s36(pretrained=args.pretrained)
            
        model.head.fc = nn.Linear(model.head.fc.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "CoAtNet" in args.FL_platform:
        
        if 'BN' in args.norm:
            print('Architecture: CoAtNet-0 with BN only')
            from timm.models.maxxvit import coatnet_bn_0_rw_224
            model = coatnet_bn_0_rw_224(pretrained=args.pretrained)

        if 'GN' in args.norm:
            print('Architecture: CoAtNet-0 with GN only')
            from timm.models.maxxvit import coatnet_bn_0_rw_224
            from utils.architectures_modifications import coatnet_to_group_norm
            model = coatnet_bn_0_rw_224(pretrained=args.pretrained)
            coatnet_to_group_norm(model)
        
        else:
            print('Architecture: CoAtNet-0')
            from timm.models.maxxvit import coatnet_0_rw_224
            model = coatnet_0_rw_224(pretrained=args.pretrained)
            
        model.head.fc = nn.Linear(model.head.fc.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "IdentityFormer" in args.FL_platform:
        print('Architecture: IdentityFormer-S36')
        from metaformer.metaformer_baselines import identityformer_s36
        model = identityformer_s36(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    elif "RandFormer" in args.FL_platform:
        print('Architecture: RandFormer-S36')
        from metaformer.metaformer_baselines import randformer_s36
        model = randformer_s36(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "RIFormer" in args.FL_platform:
        print('Architecture: RandFormer-S36')   
        from mmpretrain import get_model
        model = get_model("riformer-s36_in1k", pretrained=args.pretrained)
        model.head.fc = nn.Linear(model.head.fc.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "ResMLP" in args.FL_platform:
        print('Architecture: ResMLP-24')
        from timm.models.mlp_mixer import resmlp_24_224
        model = resmlp_24_224(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)

    elif "GMLP" in args.FL_platform:
        print('Architecture: GMLP-S16')
        from timm.models.mlp_mixer import gmlp_s16_224
        model = gmlp_s16_224(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)      

    elif "MLPMixer" in args.FL_platform:
        print('Architecture: MLPMixer-B16')
        from timm.models.mlp_mixer import mixer_b16_224
        model = mixer_b16_224(pretrained=args.pretrained)
        model.head = nn.Linear(model.head.weight.shape[1], args.num_classes)
        model.to(args.device)

    # mobile architectures
    elif 'MobileNetV3' in args.FL_platform:
        print('Architecture: MobileNet-V3-small')
        from timm.models.mobilenetv3 import mobilenetv3_small_100
        model = mobilenetv3_small_100(pretrained=args.pretrained)
        model.classifier = nn.Linear(model.classifier.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    elif "ShuffleNetV2" in args.FL_platform:
        print('Architecture: ShuffleNetV2-X1')
        import torchvision.models as torch_models
        model = torch_models.shufflenet_v2_x1_0(weights='DEFAULT' if args.pretrained else None)
        model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    elif "MobileViT" in args.FL_platform:
        print('Architecture: MobileViT-S')
        from timm.models.mobilevit import mobilevit_s
        model = mobilevit_s(pretrained=args.pretrained)
        model.head.fc = nn.Linear(model.head.fc.weight.shape[1], args.num_classes)
        model.to(args.device)
        
    name_parts = [
        args.FL_platform,
        args.dataset,
        args.split_type,
        f"run_{args.n}",
        f"lr_{args.learning_rate}",
        f"pretrained_{args.pretrained}",
        f"Seed_{args.seed}"
    ]

    if args.norm:
        name_parts.insert(1, args.norm)

    args.name_run = '_'.join(name_parts)

    if args.use_wandb:
        wandb.login()

        wandb.init(
            project="Fed_Het",
            name = args.name_run,
            config = args, 
        )

    args.output_dir = os.path.join('output', args.FL_platform, args.dataset, args.name_run)
    os.makedirs(args.output_dir, exist_ok=True)
    print_options(args, model)

    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}
    args.current_test_acc_avg = {}

    return model


