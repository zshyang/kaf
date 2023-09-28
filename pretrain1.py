'''
used in scritps/pretrain2
        scripts/pretrain3

'''
import copy
import os
from datetime import datetime

from batchgenerators.utilities.file_and_folder_operations import *
from dataset.acdc_graph import ACDC
from dataset.chd import CHD, chd_sg_collate
from experiment_log import PytorchExperimentLogger
from loss.contrast_loss import SupConLoss
from lr_scheduler import LR_Scheduler
from myconfig import get_config
# from network.unet2d import
from network.dynamic_graph_unet2d import GraphUNet2DClassification
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import *


def get_kwargs_model(args):
    model_kwargs = vars(copy.deepcopy(args))
    model_kwargs.pop('initial_filter_size')
    model_kwargs.pop('classes')
    return model_kwargs


def main():
    # initialize config
    args = get_config()

    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(
        args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = PytorchExperimentLogger(save_path, "elog", ShowTerminal=True)
    model_result_dir = join(save_path, 'model')
    maybe_mkdir_p(model_result_dir)
    args.model_result_dir = model_result_dir

    logger.print(f"saving to {save_path}")
    writer = SummaryWriter('runs/' + args.experiment_name + args.save)

    # setup cuda
    args.device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    # logger.print(f"the model will run on device {args.device}")

    # create model
    logger.print("creating model ...")
    model_kwargs = get_kwargs_model(args)
    model = GraphUNet2DClassification(
        in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes, do_instancenorm=True, **model_kwargs
    )

    if args.restart:
        logger.print('loading from saved model'+args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        save_model = dict["net"]
        model_dict.update(save_model)
        model.load_state_dict(model_dict)

    model.to(args.device)
    model = torch.nn.DataParallel(model)

    num_parameters = sum([l.nelement() for l in model.module.parameters()])
    logger.print(f"number of parameters: {num_parameters}")

    if args.dataset == 'chd':
        training_keys = os.listdir(os.path.join(args.data_dir, 'train'))
        training_keys.sort()
        train_dataset = CHD(keys=training_keys, purpose='train', args=args)
    elif args.dataset == 'acdc':
        train_dataset = ACDC(keys=list(range(1, 101)),
                             purpose='train', args=args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, drop_last=True, collate_fn=chd_sg_collate
    )

    # define loss function (criterion) and optimizer
    criterion = SupConLoss(threshold=args.slice_threshold, temperature=args.temp,
                           contrastive_method=args.contrastive_method).to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5
    )
    corr_lr = 0.001
    corr_optimizer = torch.optim.SGD(
        model.parameters(), lr=corr_lr, momentum=0.9, weight_decay=1e-5
    )

    scheduler = LR_Scheduler(
        args.lr_scheduler, args.lr, args.epochs, len(train_loader)
    )
    corr_scheduler = LR_Scheduler(
        args.lr_scheduler, corr_lr, args.epochs, len(train_loader)
    )

    for epoch in range(args.epochs):
        # train for one epoch
        train_loss = train(
            train_loader, model, criterion,
            epoch, optimizer, corr_optimizer, scheduler, corr_scheduler, logger, args
        )

        logger.print('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     .format(epoch + 1, train_loss=train_loss))

        writer.add_scalar('training_loss', train_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # save model
        save_dict = {"net": model.module.state_dict()}
        torch.save(save_dict, os.path.join(
            args.model_result_dir, "latest.pth")
        )
        if epoch % 50 == 0:
            torch.save(
                save_dict,
                os.path.join(args.model_result_dir, f"epoch_{epoch:03d}.pth")
            )


def train(
    data_loader, model, criterion, epoch, optimizer, corr_optimizer,
    scheduler, corr_scheduler, logger, args, optimizer_time=1
):
    model.train()
    cnn_loss = AverageMeter()
    graph_loss = AverageMeter()
    corr_losses = AverageMeter()
    local_graph_losses = AverageMeter()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        scheduler(optimizer, batch_idx, epoch)
        corr_scheduler(corr_optimizer, batch_idx, epoch)

        tup = {
            k: v.to(args.device)
            if isinstance(v, torch.Tensor) else v
            for k, v in tup.items()
        }

        # sometimes we drop too much data and we need to skip the batch
        if tup['keypoints_1'].shape[0] / 4 < 2:
            print('batch size too small')
            continue

        if optimizer_time == 1:
            corr_loss, f1_1, f2_1, graph_1, graph_2, mask, local_graph_1, local_graph_2 = model(
                input_dict=tup, weight_corr=args.weight_corr, weight_local=args.weight_local_contrast
            )

            bsz = f1_1.shape[0]

            features = torch.cat([f1_1.unsqueeze(1), f2_1.unsqueeze(1)], dim=1)

            graph_features = torch.cat(
                [graph_1.unsqueeze(1), graph_2.unsqueeze(1)], dim=1
            )

            if args.contrastive_method == 'pcl':
                if bsz != tup['slice_position'].shape[0]:
                    print('batch size not equal')
                    continue
                cnn_contrast_loss = criterion(
                    features, labels=tup['slice_position']
                )
                graph_scontrast_loss = criterion(
                    graph_features, labels=tup['slice_position']
                )

                cnn_loss.update(cnn_contrast_loss.item(), bsz)
                graph_loss.update(graph_scontrast_loss.item(), bsz)

                contrast_loss = args.weight_cnn_contrast * cnn_contrast_loss + \
                    args.weight_graph_contrast * graph_scontrast_loss
            elif args.contrastive_method == 'gcl':
                contrast_loss = criterion(
                    features, labels=tup['slice_position']
                )
            else:  # simclr
                contrast_loss = criterion(features)

            if torch.isnan(contrast_loss):
                print('nan contrast loss')
                continue

            corr_loss = corr_loss.mean()
            corr_losses.update(corr_loss.item(), bsz)
            corr_loss = args.weight_corr * corr_loss

            if torch.isnan(corr_loss):
                print('nan corr loss')
                continue

            loss = corr_loss + contrast_loss

            if args.weight_local_contrast > 0.:
                sum_local_graph_loss = 0.
                for mask_i, local_graph_1_i, local_graph_2_i in zip(mask, local_graph_1, local_graph_2):
                    local_graph_1_i = local_graph_1_i.permute(1, 0)
                    local_graph_2_i = local_graph_2_i.permute(1, 0)
                    local_graph_features = torch.cat(
                        [local_graph_1_i.unsqueeze(1), local_graph_2_i.unsqueeze(1)], dim=1
                    )
                    local_graph_contrast_loss = criterion(
                        local_graph_features, mask=mask_i
                    )
                    sum_local_graph_loss = sum_local_graph_loss + local_graph_contrast_loss
                mean_local_graph_contrast_loss = sum_local_graph_loss / \
                    mask.shape[0]
                local_graph_losses.update(
                    mean_local_graph_contrast_loss.item(), bsz
                )
                loss = loss + args.weight_local_contrast * mean_local_graph_contrast_loss
            else:
                local_graph_losses.update(0., bsz)

            losses.update(loss.item(), bsz)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif optimizer_time == 2:
            # optimize contrastive loss
            _, f1_1, f2_1, graph_1, graph_2 = model(tup)
            bsz = f1_1.shape[0]
            features = torch.cat([f1_1.unsqueeze(1), f2_1.unsqueeze(1)], dim=1)
            graph_features = torch.cat(
                [graph_1.unsqueeze(1), graph_2.unsqueeze(1)], dim=1
            )
            cnn_contrast_loss = criterion(
                features, labels=tup['slice_position']
            )
            graph_scontrast_loss = criterion(
                graph_features, labels=tup['slice_position']
            )

            cnn_loss.update(cnn_contrast_loss.item(), bsz)
            graph_loss.update(graph_scontrast_loss.item(), bsz)

            contrast_loss = args.weight_cnn_contrast * cnn_contrast_loss + \
                args.weight_graph_contrast * graph_scontrast_loss

            if torch.isnan(contrast_loss):
                print('nan contrast loss')
                continue
            optimizer.zero_grad()
            contrast_loss.backward()
            optimizer.step()

            # optimize corr loss
            corr_loss, _, _, _, _ = model(tup)
            corr_loss = corr_loss.mean()
            corr_losses.update(corr_loss.item(), bsz)
            corr_loss = args.weight_corr * corr_loss

            if torch.isnan(corr_loss):
                print('nan corr loss')
                continue

            corr_optimizer.zero_grad()
            corr_loss.backward()
            corr_optimizer.step()

            loss = corr_loss + contrast_loss
            losses.update(loss.item(), bsz)

        else:
            raise NotImplementedError

        logger.print(
            f"epoch:{epoch:4d}, batch:{batch_idx:4d}/{len(data_loader)}, "
            f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
            f"cnn loss:{cnn_loss.avg:.4f} "
            f"graph loss:{graph_loss.avg:.4f} "
            f"corr loss:{corr_losses.avg:.4f} "
            f"local graph loss:{local_graph_losses.avg:.4f} "
            f"total loss:{losses.avg:.4f}"
        )
    return losses.avg


if __name__ == '__main__':
    main()
