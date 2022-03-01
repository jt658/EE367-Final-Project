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


@torch.no_grad()
def validation(kShot, noiseTaskParams, device, imgDim, learner, lossFunc, psnrFunc, numInnerIterations):
    # create datasets
    val_dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=True)
    # create dataloaders & seed for reproducibility
    # Should return data set of size (numTasks, 2*kShot, numChannels=3, height, width)
    val_dataloader = DataLoader(val_dataset, batch_size=2*kShot, shuffle=True, drop_last=True)

    error = 0
    PSNR = 0

    # Iterate over tasks
    for idx, sample in enumerate(val_dataloader):
        #noiseTaskParams = [("G", sigma), ("P", lambda) ... x6]
        if idx < len(noiseTaskParams):
            noiseParam = noiseTaskParams[idx]

            # Split data into train and test partitions
            trainCleanImgs, testCleanImgs = torch.tensor_split(sample, 2, dim=0)
            trainCleanImgs.to(device)
            testCleanImgs.to(device)

            assert trainCleanImgs.shape == (10, 3, imgDim, imgDim)
            assert testCleanImgs.shape == (10, 3, imgDim, imgDim)

            # Create synthetic noisy images
            if noiseParam[0] == "G":
                # TODO: Apply gaussian noise with sigma = noiseParam[1]
                trainNoisyImgs = trainCleanImgs + sigma * torch.randn(*trainCleanImgs.shape)
                trainNoisyImgs = trainNoisyImgs.clip(torch.min(trainCleanImgs), torch.max(trainCleanImgs))

                testNoisyImgs = testNoisyImgs + sigma * torch.randn(*trainCleanImgs.shape)
                testNoisyImgs = testNoisyImgs.clip(torch.min(testCleanImgs), torch.max(testCleanImgs))
            elif noiseParam[0] == "P":
                # TODO: Apply poisson noise with parameter = noiseParam[1]
                PEAK = noiseParam[1]
                trainNoisyImgs = torch.poisson(trainCleanImgs * PEAK) / PEAK
                trainNoisyImgs = trainNoisyImgs.clip(torch.min(trainCleanImgs), torch.max(trainCleanImgs))

                testNoisyImgs = torch.poisson(testCleanImgs * PEAK) / PEAK
                testNoisyImgs = testNoisyImgs.clip(torch.min(testCleanImgs), torch.max(testCleanImgs))

            for innerIteration in range(numInnerIterations):
                trainError = lossFunc(learner(trainNoisyImgs), trainCleanImgs)
                learner.adapt(trainError)

            # Evaluate the adapted model
            predictions = learner(testNoisyImgs)
            testError = lossFunc(predictions, testCleanImgs).item()
            testPSNR = psnrFunc(predictions, testCleanImgs).item()

            # Add error and PSNR for each task
            error += testError
            PSNR += testPSNR

    # Return average error and PSNR across all tasks
    return error / len(noiseTaskParams), PSNR / len(noiseTaskParams)


def train(innerLr, outerLr, numOuterIterations, numInnerIterations, kShot, imgDim, noiseTaskParams, metaValFreq):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dncnnModel = net(in_nc=3, out_nc=3, nc=64, nb=17, act_mode='BR')
    dncnnModel.to(device)

    mamlModel = l2l.algorithms.MAML(dncnnModel, lr=outerLr)
    optimizer = optim.Adam(mamlModel.parameters(), lr=innerLr)

    # Define loss
    mseLossFunc = nn.MSELoss()

    # Define PSNR metric
    psnrFunc = PSNR().double().cuda() if device == 'cuda' else PSNR().double()

    # Keep track of PSNR and loss
    metaTrainPSNR = []
    metaTrainLoss = []
    bestAvgPSNR = 0

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
        for idx, sample in enumerate(train_dataloader):
            
            if idx < len(noiseTaskParams):
                
                noiseParam = noiseTaskParams[idx]

                # Split data into train and test partitions
                trainCleanImgs, testCleanImgs = torch.tensor_split(sample, 2, dim=0)
                trainCleanImgs.to(device)
                testCleanImgs.to(device)

                assert trainCleanImgs.shape == (10, 3, imgDim, imgDim)
                assert testCleanImgs.shape == (10, 3, imgDim, imgDim)

                # Create synthetic noisy images
                if noiseParam[0] == "G":
                    # TODO: Apply gaussian noise with sigma = noiseParam[1]
                    sigma = noiseParam[1]
                    trainNoisyImgs = trainCleanImgs + sigma * torch.randn(*trainCleanImgs.shape)
                    trainNoisyImgs = trainNoisyImgs.clip(torch.min(trainCleanImgs), torch.max(trainCleanImgs))

                    testNoisyImgs = testCleanImgs + sigma * torch.randn(*testCleanImgs.shape)
                    testNoisyImgs = testNoisyImgs.clip(torch.min(testCleanImgs), torch.max(testCleanImgs))
                elif noiseParam[0] == "P":
                    # TODO: Apply poisson noise with parameter = noiseParam[1]
                    PEAK = noiseParam[1]
                    trainNoisyImgs = torch.poisson(trainCleanImgs * PEAK) / PEAK
                    trainNoisyImgs = trainNoisyImgs.clip(torch.min(trainCleanImgs), torch.max(trainCleanImgs))

                    testNoisyImgs = torch.poisson(testCleanImgs * PEAK) / PEAK
                    testNoisyImgs = trainNoisyImgs.clip(torch.min(testCleanImgs), torch.max(testCleanImgs))

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

        # Save average error and PSNR across all tasks in the current outer iteration
        avgIterError = iterError / len(noiseTaskParams)
        metaTrainLoss.append(iterError.item() / len(noiseTaskParams))
        metaTrainPSNR.append(iterPSNR / len(noiseTaskParams))

        # Apply meta-validation for all tasks using the meta-val dataset
        # Keep track of the best model
        if outerIteration != 0 and outerIteration % metaValFreq == 0:
            metaValLoss, metaValPSNR = validation(kShot, noiseTaskParams, device, imgDim, learner, mseLossFunc,
                                                  psnrFunc, numInnerIterations)

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
