import torch
from torch.func import vmap
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
import gc


train_loss = []  # To store losses
test_loss = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data, output_to_file = True, print_epoch = 10):  
    name = data.get("name", "main")
    model = data.get("model")
    epochs = data.get("epochs")
    batchsize = data.get("batchsize")
    optimizer = data.get("optimizer")
    scheduler = data.get("scheduler")
    component_manager = data.get("component_manager")
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
            "modules": str(component_manager)
        } 
        if additional_data != None:
            params["additionalData"] = additional_data
        fp = open(params_path, "w", newline='\r\n') 
        json.dump(params, fp)
        fp.close()  


    #for temporal causality weights
    model = model.to(device)

    for epoch in range(epochs):
        model.train(True)
        train_losses = []
        for i in range(component_manager.number_of_iterations(train = True)):
            l = component_manager.apply(model, train = True)
            l.backward()    
            optimizer.step() 
            optimizer.zero_grad()

            train_losses.append(l.item())

        
                
                # Save to log file
                #log_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\n'.format(
                #    epoch, batch_idx, int(len(dataloader.dataset)/batchsize),
                #    100. * batch_idx / len(dataloader), loss.item()))
        if epoch % print_epoch ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, (i+1), component_manager.number_of_iterations(train = True),
                100. * (i+1) / component_manager.number_of_iterations(train = True), l.item()))

        train_loss.append(np.average(train_losses))

        model.eval()
        validation_losses = []
        for i in range(component_manager.number_of_iterations(train = False)):
            l = component_manager.apply(model, train = False)
            validation_losses.append(l.item())

            del l
            gc.collect()
            
        if epoch % print_epoch ==0:
            print('Validation Epoch: {} \tLoss: {:.10f}'.format(
                epoch, np.average(validation_losses)))
            
        
        test_loss.append(np.average(validation_losses))
                
        if output_to_file and epoch % 100 == 0:
            epoch_path = os.path.join(model_dir, f"model_{epoch}.pt")
            torch.save(model, epoch_path)
        
        if scheduler != None:
            scheduler.step()

        torch.cuda.empty_cache()
    
    # Save the model
    if output_to_file:
        torch.save(model, model_path)
        
        plt.plot(train_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'{output_dir}/training_loss.png')
        plt.clf()
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(f'{output_dir}/train_and_test_loss.png')
        plt.clf()
        label = ["Residual loss", "IC loss"]
        residual_losses = component_manager.search("Residual", train = False)
        for i in range(len(residual_losses)):
            j = 0
            for single_loss in residual_losses[i].loss:
                plt.plot(single_loss.history)
                label.append("Residual_loss_"+str(i)+"_"+str(j))
                j += 1
        ic_losses = component_manager.search("IC", train = False)
        for i in range(len(ic_losses)):
            j = 0
            for single_loss in ic_losses[i].loss:
                plt.plot(single_loss.history)
                label.append("IC_loss_"+str(i)+"_"+str(j))
                j += 1
        plt.legend(label)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.savefig(f'{output_dir}/train_losses.png')
        plt.show()

    return np.min(test_loss)