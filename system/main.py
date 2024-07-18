import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from torchvision import models

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import BiLSTM_TextClassification
from flcore.trainmodel.alexnet import alexnet
from system.flcore.trainmodel.U_Net import UNet
from system.flcore.trainmodel.cbam_binglian_lkRelu_upchan import cbam_binglian_lkRelu_upchan
from system.flcore.trainmodel.cbam_binglian_lkRelu import cbam_binglian_lkRelu
from system.flcore.trainmodel.cbam import cbam
from system.flcore.trainmodel.cbam_binglian import cbam_binglian
from system.flcore.trainmodel.cbam_lkRelu import cbam_lkRelu
from system.flcore.trainmodel.cbam_upchan import cbam_upchan
from system.flcore.trainmodel.cbam_upchan_lkRelu import cbam_upchan_lkRelu
from system.flcore.trainmodel.cbam_zhongjian import cbam_zhongjian
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(1)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)
        elif model_str == "trans":
            if args.dataset == "tumor":
                args.model = ViT(image_size=224, patch_size=args.batch_size, num_classes=args.num_classes, dim=1024,
                                 depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(args.device)
        elif model_str == "cnn":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset == "omniglot":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif args.dataset == "cifar10" or args.dataset == "cifar100":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
                # args.model = CnnAtt(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif args.dataset == "Digit5":
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=179776).to(args.device)

        elif model_str == "dnn":  # non-convex
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "unet":
            import torch.nn as nn
            model = UNet()
            args.model = model.to(args.device)
        elif model_str == "cbam":  # 116s
            import torch.nn as nn
            model = cbam(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
            args.model = model.to(args.device)

        elif model_str == "cbam_zhongjian":  # 120s
            import torch.nn as nn
            model = cbam_zhongjian(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
            args.model = model.to(args.device)

        elif model_str == "cbam_upchan":  # 116s
            import torch.nn as nn
            model = cbam_upchan(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
            args.model = model.to(args.device)

        elif model_str == "cbam_lkRelu":  # 120s
            import torch.nn as nn
            model = cbam_lkRelu(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
            args.model = model.to(args.device)

        elif model_str == "cbam_upchan_lkRelu":  #
            import torch.nn as nn
            model = cbam_upchan_lkRelu(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
            args.model = model.to(args.device)

        elif model_str == "cbam_binglian":  # 123s
            import torch.nn as nn
            model = cbam_binglian(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
            args.model = model.to(args.device)

        elif model_str == "cbam_binglian_lkRelu":  #
            import torch.nn as nn
            model = cbam_binglian_lkRelu(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
            args.model = model.to(args.device)

        elif model_str == "cbam_binglian_lkRelu_upchan":  # butong
            import torch.nn as nn
            model = cbam_binglian_lkRelu_upchan(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
            args.model = model.to(args.device)

        elif model_str == "res18":
            import torch.nn as nn
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
            args.model = model.to(args.device)

        elif model_str == "res50":  # 118s
            import torch.nn as nn
            model = models.resnet50(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
            args.model = model.to(args.device)
        elif model_str == "densenet":  # 284s
            import torch.nn as nn
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
            for param in model.parameters():
                param.requires_grad = True
            args.model = model.to(args.device)
        elif model_str == "mobile_v3":  # 54.7s
            import torch.nn as nn
            model = models.mobilenet_v3_small(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.num_classes)
            args.model = model.to(args.device)
        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "vgg16":
            import torch.nn as nn
            model = models.vgg16(pretrained=True)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, args.num_classes)
            #   model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
            for param in model.parameters():
                param.requires_grad = True
            args.model = model.to(args.device)

        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = models.mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=hidden_dim,
                                                   output_size=args.num_classes,
                                                   num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                                   attention_dropout=0,
                                                   embedding_length=hidden_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=hidden_dim, max_len=max_len, vocab_size=vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=hidden_dim, nhead=2, d_hid=hidden_dim, nlayers=2,
                                          num_classes=args.num_classes).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset,
                 algorithm=args.algorithm,
                 goal=args.goal,
                 times=args.times,
                 length=args.global_rounds / args.eval_gap + 1)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")  # 实验目标，等于备注
    parser.add_argument('-data', "--dataset", type=str, default="minibal")  # 数据集
    parser.add_argument('-nb', "--num_classes", type=int, default=4)  # 输出类别!!!
    parser.add_argument('-m', "--model", type=str, default="cbam_binglian_lkRelu_upchan")  # model
    parser.add_argument('-p', "--head", type=str, default="cnn")  #cnn
    parser.add_argument('-lbs', "--batch_size", type=int, default=2)  # 批次
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)  # 全局轮数，也算communication rounds
    parser.add_argument('-ls', "--local_steps", type=int, default=1)  # 本地步数,也算局部epoch
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")  # 联邦算法: 头文件里面，11行开始
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")  # 每轮客户比例
    parser.add_argument('-nc', "--num_clients", type=int, default=3,
                        help="Total number of clients")  # 客户总数
    parser.add_argument('-dev', "--device", type=str, default="mps",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")  # 上一个运行时间
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")  # 每轮客户随机比例
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")  # 评价轮差
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")  # 差分隐私
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")  # 训练中客户离线比例
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")  # 培训时慢客户比例
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")  # 发送全局模型时速度慢的客户端的速率
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")  # 是否根据时间成本对每轮客户进行分组和选择
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument('-ta', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=1)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable,but mps could run\n")
        args.device = "mps"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")