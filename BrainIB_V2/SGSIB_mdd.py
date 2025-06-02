import os
import os.path as osp
import argparse

import torch
import torch.nn.functional as F

import numpy as np

from torch_geometric.loader import DataLoader

from SGSIB.GNN import GNN
from SGSIB.sub_graph_generator import MLP_subgraph
from SGSIB.utils import train, test

from data.dataset import BrIB_RESTfMRIDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDD SGSIB')
    parser.add_argument('--iters_per_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mi_weight', type=float, default=0.001)
    parser.add_argument('--pos_weight', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_learning_rate', type=float, default=0.0005)
    parser.add_argument('--SGmodel_learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    

    # Split into train/dev using metadata-based logic
    train_dataset = BrIB_RESTfMRIDataset(
         metadata_path="../../Data/metadata.csv",
        data_dir="../../Data/fMRI/AAL",
        split="train"
    )

    dev_dataset = BrIB_RESTfMRIDataset(
        metadata_path="../../Data/metadata.csv",
        data_dir="../../Data/fMRI/AAL",
        split="dev"
    )
    train_dataset = [data for data in train_dataset]
    dev_dataset = [data for data in dev_dataset]

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    num_node_features = 116
    num_edge_features = 1

    model = GNN(num_of_features=num_node_features, device=device).to(device)
    SG_model = MLP_subgraph(node_features_num=num_node_features, edge_features_num=num_edge_features, device=device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.model_learning_rate},
        {'params': SG_model.parameters(), 'lr': args.SGmodel_learning_rate}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        avg_loss, mi_loss = train(args, model, train_dataset, optimizer, epoch, SG_model, device)
        acc_train, acc_dev, dev_loss = test(args, model, train_dataset, dev_dataset, SG_model, device)

        print(f"Epoch {epoch}: Train Acc = {acc_train:.4f}, Dev Acc = {acc_dev:.4f}, Loss = {avg_loss:.4f}, MI Loss = {mi_loss:.4f}")

        # Save checkpoints
        savedir = "./SGSIB/model"
        os.makedirs(savedir, exist_ok=True)

        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, osp.join(savedir, f"GNN_epoch{epoch}.tar"))
        torch.save({"epoch": epoch, "state_dict": SG_model.state_dict()}, osp.join(savedir, f"subgraph_epoch{epoch}.tar"))

        with open(osp.join(savedir, "log.txt"), 'a+') as f:
            f.write(f"{epoch} {avg_loss:.4f} {acc_train:.4f} {acc_dev:.4f} {mi_loss:.4f}\n")

        scheduler.step()
        torch.cuda.empty_cache()
        


    print("\n✅ Finished training with custom dataset and train/dev split.")
