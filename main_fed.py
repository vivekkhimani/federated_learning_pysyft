import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
from collections import defaultdict

from utils import Arguments
from models import CNN

hook = sy.TorchHook(torch)  #extra functionality to support FL

def virtualWorkers():
    one=sy.VirtualWorker(hook, id="one")
    two=sy.VirtualWorker(hook, id="two")
    three=sy.VirtualWorker(hook, id="three")
    four=sy.VirtualWorker(hook, id="four")
    five=sy.VirtualWorker(hook, id="five")
    six=sy.VirtualWorker(hook, id="six")
    seven=sy.VirtualWorker(hook, id="seven")
    eight=sy.VirtualWorker(hook, id="eight")
    nine=sy.VirtualWorker(hook, id="nine")
    ten=sy.VirtualWorker(hook, id="ten")
    eleven=sy.VirtualWorker(hook, id="eleven")
    twelve=sy.VirtualWorker(hook, id="twelve")
    thirteen=sy.VirtualWorker(hook, id="thirteen")
    fourteen=sy.VirtualWorker(hook, id="fourteen")
    fifteen=sy.VirtualWorker(hook, id="fifteen")
    sixteen=sy.VirtualWorker(hook, id="sixteen")
    seventeen=sy.VirtualWorker(hook, id="seventeen")
    eighteen=sy.VirtualWorker(hook, id="eighteen")
    nineteen=sy.VirtualWorker(hook, id="nineteen")
    twenty=sy.VirtualWorker(hook, id="twenty")
    twenty_one=sy.VirtualWorker(hook, id="twenty_one")
    twenty_two=sy.VirtualWorker(hook, id="twenty_two")
    twenty_three=sy.VirtualWorker(hook, id="twenty_three")
    twenty_four=sy.VirtualWorker(hook, id="twenty_four")
    twenty_five=sy.VirtualWorker(hook, id="twenty_five")
    twenty_six=sy.VirtualWorker(hook, id="twenty_six")
    twenty_seven=sy.VirtualWorker(hook, id="twenty_seven")
    twenty_eight=sy.VirtualWorker(hook, id="twenty_eight")
    twenty_nine=sy.VirtualWorker(hook, id="twenty_nine")
    thirty=sy.VirtualWorker(hook, id="thirty")
    thirty_one=sy.VirtualWorker(hook, id="thirty_one")
    thirty_two=sy.VirtualWorker(hook, id="thirty_two")

    return [one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen,fourteen,fifteen,sixteen,seventeen,eighteen,nineteen,twenty,twenty_one,twenty_two,twenty_three,twenty_four,twenty_five,twenty_six,twenty_seven,twenty_eight,twenty_nine,thirty,thirty_one,thirty_two]

vList = virtualWorkers()

def loadMNISTData():
    federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])).federate((vList[0],vList[1],vList[2],vList[3],vList[4],vList[5],vList[6],vList[7],vList[8],vList[9],vList[10],vList[11],vList[12],vList[13],vList[14],vList[15],vList[16],vList[17],vList[18],vList[19],vList[20],vList[21],vList[22],vList[23],vList[24],vList[25],vList[26],vList[27],vList[28],vList[29],vList[30],vList[31])),batch_size=Arguments.args.batch_size, shuffle=True, **Arguments.kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),batch_size=Arguments.args.test_batch_size, shuffle=True, **Arguments.kwargs)

    return federated_train_loader,test_loader


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # <-- now it is a distributed dataset
        model.send(data.location) # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss(output, target)
        loss.backward()
        optimizer.step()
        model.get() # <-- NEW: get the model back
        if batch_idx % args.log_interval == 0:
            loss = loss.get() # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size, #batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def launch():
    model = CNN.CNN().to(Arguments.device)
    optimizer = optim.SGD(model.parameters(), lr=Arguments.args.lr)
    train_loader,test_loader = loadMNISTData()

    for epoch in range(1,Arguments.args.epochs+1):
        train(Arguments.args,model,Arguments.device,train_loader,optimizer,epoch)
        test(Arguments.args,model,Arguments.device,test_loader)

    if(Arguments.args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

##DRIVER##
if __name__ == '__main__':
    launch()
