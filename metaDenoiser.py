import numpy as np
import os
import logging
import torch
from torch import nn, optim
from torch.nn import functional as F
import learn2learn as l2l
import matplotlib.pyplot as plt
from BSDS300Dataset import BSDS300Dataset
from piqa import PSNR
from network_dncnn import DnCNN as net
from torch.utils.data import DataLoader
import time 


def validation(kShot, noiseTaskParams, device, imgDim, learner, lossFunc, psnrFunc, numInnerIterations):
    # create datasets
    val_dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=True)
    # create dataloaders & seed for reproducibility
    # Should return data set of size (numTasks, 2*kShot, numChannels=3, height, width)
    val_dataloader = DataLoader(val_dataset, batch_size=2*kShot, shuffle=True, drop_last=True)

    error = 0
    PSNR = 0
    dynRange = [0.0, 1.0]

    print("Meta-Val PSNRs Per Task:")

    # Iterate over tasks
    for idx, sample in enumerate(val_dataloader):

        sample = sample.to(device)

        #noiseTaskParams = [("G", sigma), ("P", lambda) ... x6]
        if idx < len(noiseTaskParams):
            noiseParam = noiseTaskParams[idx]

            # Split data into train and test partitions
            trainCleanImgs, testCleanImgs = torch.tensor_split(sample, 2, dim=0)
            #trainCleanImgs = trainCleanImgs.to(device)
            #testCleanImgs = testCleanImgs.to(device)

            assert trainCleanImgs.shape == (10, 3, imgDim, imgDim)
            assert testCleanImgs.shape == (10, 3, imgDim, imgDim)

            # Create synthetic noisy images
            if noiseParam[0] == "G":
                # TODO: Apply gaussian noise with sigma = noiseParam[1]
                sigma = noiseParam[1]
                trainNoisyImgs = trainCleanImgs + sigma * torch.randn(*trainCleanImgs.shape).cuda()
                trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                testNoisyImgs = testCleanImgs + sigma * torch.randn(*testCleanImgs.shape).cuda()
                testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])

            elif noiseParam[0] == "P":
                # TODO: Apply poisson noise with parameter = noiseParam[1]
                PEAK = noiseParam[1]
                trainNoisyImgs = torch.poisson(trainCleanImgs * PEAK) / PEAK
                trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                testNoisyImgs = torch.poisson(testCleanImgs * PEAK) / PEAK
                testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])

            for innerIteration in range(numInnerIterations):
                trainError = lossFunc(learner(trainNoisyImgs), trainCleanImgs)
                learner.adapt(trainError)

            # Evaluate the adapted model
            predictions = learner(testNoisyImgs)
            predictions = predictions.clip(0.0, 1.0)
            testError = lossFunc(predictions, testCleanImgs).item()
            testPSNR = psnrFunc(predictions, testCleanImgs).item()

            print(f'Noise Params: {noiseParam[1]} Meta-Val Task PSNR: {testPSNR:.04f} Noisy Image PSNR: {psnrFunc(testNoisyImgs, testCleanImgs):.04f}')     

            #fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,12))
            #ax[0].imshow(torch.permute(torch.squeeze(testCleanImgs[0,:,:,:]), (1,2,0)).cpu().detach().numpy())
            #ax[0].set_title("Clean Image")

            #ax[1].imshow(torch.permute(torch.squeeze(testNoisyImgs[0,:,:,:]), (1,2,0)).cpu().detach().numpy())
            #ax[1].set_title("Noisy Image")

            #ax[2].imshow(torch.permute(torch.squeeze(predictions[0,:,:,:]), (1,2,0)).cpu().detach().numpy())
            #ax[2].set_title("Denoised Image") 

            # Add error and PSNR for each task
            error += testError
            PSNR += testPSNR

    # Return average error and PSNR across all tasks
    return error / len(noiseTaskParams), PSNR / len(noiseTaskParams)


def train(innerLr, outerLr, numOuterIterations, numInnerIterations, kShot, imgDim, noiseTaskParams, metaValFreq):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(device)

    dncnnModel = net(in_nc=3, out_nc=3, nc=64, nb=17, act_mode='BR', use_bias=True)
    dncnnModel.to(device)

    mamlModel = l2l.algorithms.MAML(dncnnModel, lr=innerLr)
    optimizer = optim.Adam(mamlModel.parameters(), lr=outerLr)

    # Define loss
    mseLossFunc = nn.MSELoss() 

    # Define PSNR metric
    psnrFunc = PSNR().double().cuda()

    # Keep track of PSNR and loss
    metaTrainPSNR = []
    metaTrainLoss = []
    bestAvgPSNR = 0
    dynRange = [0.0, 1.0]

    for outerIteration in range(numOuterIterations):

        optimizer.zero_grad()

        # create datasets
        # (numberPatches, 2, height, width)
        train_dataset = BSDS300Dataset(patch_size=32, split='train', use_patches=True)
        # create dataloaders & seed for reproducibility
        # Should return data set of size (numTasks, 2*kShot, numChannels=3, height, width)
        train_dataloader = DataLoader(train_dataset, batch_size=2 * kShot, shuffle=True, drop_last=True)

        iterError = 0.0
        iterPSNR = 0.0

        # Iterate over tasks
        # {"G": sigma, "P": ...}
        iterTimeStart = time.time()
        for idx, sample in enumerate(train_dataloader):

            sample = sample.to(device)

            if idx < len(noiseTaskParams):
                
                noiseParam = noiseTaskParams[idx]
                # Split data into train and test partitions
                trainCleanImgs, testCleanImgs = torch.tensor_split(sample, 2, dim=0)
                #trainCleanImgs = trainCleanImgs.to(device)
                #testCleanImgs = testCleanImgs.to(device)

                assert trainCleanImgs.shape == (10, 3, imgDim, imgDim)
                assert testCleanImgs.shape == (10, 3, imgDim, imgDim)
                # Create synthetic noisy images
                if noiseParam[0] == "G":
                    # TODO: Apply gaussian noise with sigma = noiseParam[1]
                    sigma = noiseParam[1]
                    trainNoisyImgs = trainCleanImgs + sigma * torch.randn(*trainCleanImgs.shape).cuda()
                    trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                    testNoisyImgs = testCleanImgs + sigma * torch.randn(*testCleanImgs.shape).cuda()
                    testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])

                elif noiseParam[0] == "P":
                    # TODO: Apply poisson noise with parameter = noiseParam[1]
                    PEAK = noiseParam[1]
                    trainNoisyImgs = torch.poisson(trainCleanImgs * PEAK) / PEAK
                    trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                    testNoisyImgs = torch.poisson(testCleanImgs * PEAK) / PEAK
                    testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])

                learner = mamlModel.clone()
                for innerIteration in range(numInnerIterations):
                    trainError = mseLossFunc(learner(trainNoisyImgs), trainCleanImgs)
                    learner.adapt(trainError)

                # Evaluate the adapted model
                predictions = learner(testNoisyImgs)
                predictions = predictions.clip(0.0, 1.0)    # <-- psnrFunc() fails if values outside of dynamic range
                testError = mseLossFunc(predictions, testCleanImgs)
                testPSNR = psnrFunc(predictions, testCleanImgs).item()

                # Add error and PSNR for each task
                iterError += testError
                iterPSNR += testPSNR
        iterTimeEnd = time.time()
        iterTime = iterTimeEnd-iterTimeStart
        # Save average error and PSNR across all tasks in the current outer iteration
        avgIterError = iterError / len(noiseTaskParams)
        metaTrainLoss.append(iterError.item() / len(noiseTaskParams))
        metaTrainPSNR.append(iterPSNR / len(noiseTaskParams))

        print(f'Iteration: {outerIteration} | Meta-Train Loss: {metaTrainLoss[-1]:.04f} | Meta-Train PSNR: {metaTrainPSNR[-1]:.04f} | Time: {iterTime:.04f}')

        # Apply meta-validation for all tasks using the meta-val dataset
        # Keep track of the best model
        if outerIteration != 0 and outerIteration % metaValFreq == 0:
            metaValLoss, metaValPSNR = validation(kShot, noiseTaskParams, device, imgDim, learner, mseLossFunc,
                                                  psnrFunc, numInnerIterations)

            print(f'Meta-Val Iteration: {outerIteration} | Meta-Val Loss: {metaValLoss:.04f} | Meta-Val PSNR: {metaValPSNR:.04f}')

            # Save parameters of inner model with the best result
            if metaValPSNR > bestAvgPSNR:
                bestAvgPSNR = metaValPSNR
                torch.save(learner.state_dict(), "best-model.pth.tar")

        # Update outer loop parameters/ meta-learning parameters
        avgIterError.backward()
        optimizer.step()

def test(model, cleanTestData, noisyTestData, numInnerIterations):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Split data into partition for fine tuning and partition for final results
    # TODO: Figure out if my dimensions are correct
    fineTuneDataClean = cleanTestData[0, ...]
    fineTuneDataNoisy = noisyTestData[0, ...]

    testDataClean = cleanTestData[1:, ...]
    testDataNoisy = noisyTestData[1:, ...]

    learner = model.clone()

    # Define loss
    mseLossFunc = nn.MSELoss()
    # Define PSNR metric
    psnrFunc = PSNR().double().cuda() if device == 'cuda' else PSNR().double()

    for innerIteration in range(numInnerIterations):
        trainError = mseLossFunc(learner(fineTuneDataNoisy), fineTuneDataClean)
        learner.adapt(trainError)

    # Evaluate the adapted model
    predictions = learner(testDataNoisy)
    # TODO: Maybe normalize both of these across the number of images in the test set?
    testError = mseLossFunc(predictions, testDataClean)
    testPSNR = psnrFunc(predictions, testDataClean).item()