import torch
from torch.func import vmap
import numpy as np
from pinns_v2.loss import loss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
import gc


all_train_losses = []
train_losses = []  # To store losses
test_losses = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data, output_to_file = True):  
    name = data.get("name", "main")
    model = data.get("model")
    epochs = data.get("epochs")
    batchsize = data.get("batchsize")
    optimizer = data.get("optimizer")
    scheduler = data.get("scheduler")
    pde_fn = data.get("pde_fn")
    ic_fns = data.get("ic_fns")
    eps_time = data.get("eps_time")
    domaindataset = data.get("domain_dataset")
    icdataset = data.get("ic_dataset")
    superviseddataset = data.get("supervised_dataset")
    validationdomaindataset = data.get("validation_domain_dataset")
    validationicdataset = data.get("validation_ic_dataset")
    additional_data = data.get("additional_data")

    if output_to_file:
        current_file = os.getcwd()
        output_dir = os.path.join(current_file, "output", name)
        
        if os.path.exists(output_dir):
            counter = 1
            while True:
                output_dir = os.path.join(current_file, "output", name+"_"+str(counter))
                if not os.path.exists(output_dir):
                    break
                else:
                    counter +=1
                    
        model_dir = os.path.join(output_dir, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            

        model_path = os.path.join(model_dir, f"model.pt")
        file_path = f"{output_dir}/train.txt"

        params_path = f"{output_dir}/params.json"
        params = {
            "name": name,
            "model": str(model),
            "epochs": epochs,
            "batchsize": batchsize,
            "optimizer": str(optimizer),
            "scheduler": str(scheduler.state_dict()) if scheduler!=None else "None",
            "domainDataset": str(domaindataset),
            "supervisedDataset": str(superviseddataset),
            "icDataset": str(icdataset),
            "validationDomainDataset": str(validationdomaindataset),
            "validationICDataset": str(validationicdataset)
        } 
        if additional_data != None:
            params["additionalData"] = additional_data
        fp = open(params_path, "w", newline='\r\n') 
        json.dump(params, fp)
        fp.close()  

    residual_losses = []
    ic_losses = [[] for i in range(len(ic_fns))]

    #for temporal causality weights
    model = model.to(device)

    dd = iter(domaindataset)
    icd = iter(icdataset)

    dsd = None
    if superviseddataset != None:
        dsd = iter(superviseddataset)

    vdd = None
    vicd = None
    if validationdomaindataset != None and validationicdataset != None:
        vdd = iter(validationdomaindataset)
        vicd = iter(validationicdataset)

    for epoch in range(epochs):
        model.train(True)
        epoch_losses = []
        for i in range(len(domaindataset)):
            x_in = next(dd)          
            x_ic = next(icd)
            
            x_in = torch.Tensor(x_in).to(device)           
            x_ic = torch.Tensor(x_ic).to(device)
            
            x_dsd = None
            if dsd != None:
                x_dsd = next(dsd)
                x_dsd = torch.Tensor(x_dsd).to(device)
            
            l = loss(model, pde_fn, ic_fns, eps_time, x_in, x_ic, x_dsd = x_dsd)
            l.backward()    
            optimizer.step() 
            optimizer.zero_grad()

            if i % 10 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, i, len(domaindataset),
                    100. * i / len(domaindataset), l.item()))
                
                # Save to log file
                #log_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\n'.format(
                #    epoch, batch_idx, int(len(dataloader.dataset)/batchsize),
                #    100. * batch_idx / len(dataloader), loss.item()))
                
            all_train_losses.append(l.item())  # Storing the loss
            epoch_losses.append(l.item())

            del x_in, x_ic, x_dsd, l
            gc.collect()

        torch.cuda.empty_cache()

        if validationicdataset != None and validationdomaindataset != None:
            model.eval()
            validation_losses = []
            for i in range(len(domaindataset)):
                x_in = next(vdd)
                x_ic = next(vicd)
                x_in = torch.Tensor(x_in).to(device)
                x_ic = torch.Tensor(x_ic).to(device)

                l = loss(model, pde_fn, ic_fns, eps_time, x_in, x_ic)
                validation_losses.append(l.item())

                del x_in, x_ic, l
                gc.collect()

            print('Validation Epoch: {} \tLoss: {:.10f}'.format(
                    epoch, np.average(validation_losses)))
            #log_file.write('Validation Epoch: {} \tLoss: {:.10f}'.format(epoch, np.average(validation_losses)))
            test_losses.append(np.average(validation_losses))
                
        if output_to_file and epoch % 20 == 0:
            epoch_path = os.path.join(model_dir, f"model_{epoch}.pt")
            torch.save(model, epoch_path)
        
        if scheduler != None:
            scheduler.step()
        train_losses.append(np.average(epoch_losses))

        torch.cuda.empty_cache()
    
    # Save the model
    if output_to_file:
        torch.save(model, model_path)
        
        plt.plot(all_train_losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'{output_dir}/training_loss.png')
        plt.clf()
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(f'{output_dir}/test_loss.png')
        plt.clf()
        label = ["Residual loss"]
        plt.plot(residual_losses)
        for i in range(len(ic_fns)):
            plt.plot(ic_losses[i])
            label.append("IC_loss_"+str(i))
        plt.legend(label)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.savefig(f'{output_dir}/train_losses.png')
        plt.show()

    return np.min(test_losses)
