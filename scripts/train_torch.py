import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from audtorch.metrics.functional import pearsonr

batch_size = 4096
num_neurons = 299

from modeling.models.SCNN import ImageDataset, net_one_neuron

if __name__ == "__main__":

    y_all_train = np.load('../Rsp.npy')
    y_all_val = np.load('../valRsp.npy')

    x_train = np.load('../train_x.npy')
    x_val = np.load('../val_x.npy')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])
    #net = nn.DataParallel(net)
    net.to(device)
    Imageset = ImageDataset(x_train, y_all_train)
    loader = DataLoader(Imageset, batch_size=batch_size, shuffle=True)
    valset = ImageDataset(x_val, y_all_val)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08)
    optimizers = [torch.optim.Adam(sub_net.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-08) for sub_net in net]
    criterion = torch.nn.L1Loss(reduction='none')
    #optimizer.load_state_dict(torch.load('optimizer_model'))
    #net.load_state_dict(torch.load('model_result'))
    num_epochs = 50
    all_loss = []
    all_corr = []
    best_corr = np.full(num_neurons, -100.1, dtype=float)
    best_model = [[] for i in range(num_neurons)]
    run_flag = [True]*num_neurons
    for epoch in tqdm(range(num_epochs)):
        #torch.save(net.state_dict(), "model_result_seperated_299")
        # Train
        for subnet in net:
            subnet.train()
        avg_loss = 0.0
        for batch_num, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            for i, (subnet,optimizer) in enumerate(zip(net,optimizers)):
                if(run_flag[i] == False):
                    continue
                optimizer.zero_grad()
                output = subnet(x)
                output = torch.reshape(output,(output.shape[0],))
                y_neuron = y[:,i]
                #y_exp = torch.exp(y_neuron)
                # loss = criterion(output,y_neuron)
                # loss_w = torch.mean(loss*y_exp)
                #loss = torch.mean(loss)
                loss = -pearsonr(output,y_neuron,batch_first=True) + 0.1*torch.mean(torch.abs(y_neuron)) + 0.1*torch.mean(criterion(output,y_neuron))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= num_neurons

        # Validate
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.eval()
            num_correct = 0
            test_loss = 0
            prediction = []
            actual = []
            for batch_num, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                outputs = torch.stack([subnet(x) for subnet in net] )
                outputs = torch.reshape(outputs, (outputs.shape[0], outputs.shape[1]))
                outputs = torch.transpose(outputs,0,1)
                # y_exp = torch.exp(y)
                # loss = criterion(outputs, y)
                # #loss = torch.mean(loss)
                # loss_w = torch.mean(loss * y_exp + 0.001* torch.abs(y))
                loss = torch.mean(-pearsonr(outputs, y, batch_first=False))+ 0.1*torch.mean(torch.abs(y)) + 0.1*torch.mean(criterion(outputs,y))
                test_loss += loss.item()
                prediction.extend(outputs.cpu().numpy())
                actual.extend(y.cpu().numpy())
            test_loss /= (len(valset) / 128)
            # scheduler.step(test_loss)
            torch.cuda.empty_cache()
            all_loss.append(test_loss)

            prediction = np.stack(prediction)
            actual = np.stack(actual)

            R = np.zeros(num_neurons)
            for neuron in range(num_neurons):
                pred1 = prediction[:, neuron]
                val_y = actual[:, neuron]
                y_arg = np.argsort(val_y)

                u2 = np.zeros((2, 1000))
                u2[0, :] = np.reshape(pred1, (1000))
                u2[1, :] = np.reshape(val_y, (1000))

                c2 = np.corrcoef(u2)
                R[neuron] = c2[0, 1]
            all_corr.append(R)

            for neuron in range(num_neurons):
                if epoch > 50 and np.mean(np.stack(all_corr[epoch-5:epoch])[neuron]) - np.mean(np.stack(all_corr[epoch-40: epoch-35])[neuron]) < 0:
                    run_flag[neuron] = False
                if all_corr[epoch][neuron] > best_corr[neuron]:
                    best_model[neuron] = net[neuron].state_dict()
                    best_corr[neuron] = all_corr[epoch][neuron]

            for i, subnet in enumerate(net):
                subnet.load_state_dict(best_model[i])
            torch.save(net.state_dict(), "model_test_corr_mae")
            print('Epoch: {}, test loss: {}, corr: {}'.format(epoch, test_loss, np.average(R)))

    np.save("all_corr_change_corr+mae", np.stack(all_corr))
    np.save("loss_data_corr+mae",np.stack(all_loss))

