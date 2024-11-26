import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from model.hybrid import WMCL_HN
from Other.dataset import MultiModalDataset
from Other.load_data import data_load
from Main.train_val import train_val
from Main.test import test

# Load data
path = r'/data'
mri1, pet1, csf1, gnd1 = data_load(path=path, str='mync')
mri2, pet2, csf2, gnd2 = data_load(path=path, str='myad')
mri, pet, csf, gnd = [np.concatenate((data1, data2), axis=0)
                      for data1, data2 in zip((mri1, pet1, csf1, gnd1), (mri2, pet2, csf2, gnd2))]

# Parameters
num_epochs = 250
batch_size_train = 20
batch_size_val = 20

# Lists to store results
results = []

# Main loop
for seed in [2, 4, 6, 8, 10]:

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # Use seed for reproducibility

    for fold1, (train_index, test_index) in enumerate(skf.split(mri, gnd)):
        print(f"Seed {seed}, Fold1 {fold1 + 1}")
        model = []

        # Split data
        x_train_val_mri, x_test_mri = mri[train_index], mri[test_index]
        x_train_val_pet, x_test_pet = pet[train_index], pet[test_index]
        x_train_val_csf, x_test_csf = csf[train_index], csf[test_index]
        y_train_val, y_test = gnd[train_index], gnd[test_index]

        test_dataset = MultiModalDataset(mri=x_test_mri, pet=x_test_pet, csf=x_test_csf, labels=y_test)
        test_loader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)

        for fold, (train_index, val_index) in enumerate(skf.split(x_train_val_mri, y_train_val)):
            print(f"Seed {seed}, Fold1 {fold1 + 1}, Fold2 {fold + 1}")

            # Split training and validation data
            x_train_mri, x_val_mri = x_train_val_mri[train_index], x_train_val_mri[val_index]
            x_train_pet, x_val_pet = x_train_val_pet[train_index], x_train_val_pet[val_index]
            x_train_csf, x_val_csf = x_train_val_csf[train_index], x_train_val_csf[val_index]
            y_train, y_val = y_train_val[train_index], y_train_val[val_index]

            # Create datasets and dataloaders
            train_dataset = MultiModalDataset(mri=x_train_mri, pet=x_train_pet, csf=x_train_csf, labels=y_train)
            val_dataset = MultiModalDataset(mri=x_val_mri, pet=x_val_pet, csf=x_val_csf, labels=y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)


            # Initialize and train the model
            HyNet = WMCL_HN().cuda()
            optimizer = optim.SGD(HyNet.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
            train_val(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, model=HyNet)
            HyNet.load_state_dict(torch.load('checkpoint.pt'))
            model.append(HyNet)

        # Test model
        result = test(test_loader=test_loader, model=model)
        results.append(result)

        # Save results to Excel
        df = pd.DataFrame(results)
        df.to_excel("result.xlsx", index=False)
