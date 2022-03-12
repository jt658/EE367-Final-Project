# Citations:
# We would like to cite the following GitHub repo as a reference for our code:
# https://github.com/edwin-pan/MetaHDR
# Authors: Edwin Pan and Anthony Vento


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
from SIDDDataset import SIDDDataset
from PIL import Image

'''
Meta-validation function: Tracks performance on the Meta-Denoiser model's ability to denoise various levels and types of noisy images

Parameters:
kShot:              Number of image patches to learn on for each task

noiseTaskParams:    List of tuples. Each tuple (p1, p2) represents the noise parameters for a task where p1 = "G" for Gaussian denoising 
                    and p1 = "P" for Poisson denoising. p2 corresponding to the noise level. For Gaussian denoising p2 = standard deviation (sigma).
                    For Poisson denoising p2 = scaling factor (beta).

imgDim:             Spatial dimensions of the image patches 

learner:            MAML model from l2l

lossFunc:           Loss function for the model

psnrFunc:           PSNR function for the model 

numInnerIterations: Number of inner iterations to use when adapting the DnCNN model

use_patches:        Boolean that indicates whether to validate on patches or whole images

visualizeImgs:      Boolean that indicates whether or not to plot resulting clean, noisy, and denoised image patches for each task

reshape:            Boolean that indicates whether or not to downside whole images 
'''
def validation(kShot, noiseTaskParams, device, imgDim, learner, lossFunc, psnrFunc, numInnerIterations, use_patches=True, visualizeImgs=False, reshape=False):
    # create datasets and dataloader for the validation data 
    val_dataset = BSDS300Dataset(patch_size=imgDim, split='validation', use_patches=use_patches, reshape=reshape)
    # Should return data set of size (numTasks, 2*kShot, numChannels=3, height, width)
    val_dataloader = DataLoader(val_dataset, batch_size=2*kShot, shuffle=True, drop_last=True)

    error = 0
    PSNR = 0
    dynRange = [0.0, 1.0]

    print("Meta-Val PSNRs Per Task:")

    # Iterate over tasks
    for idx, sample in enumerate(val_dataloader):
        
        # Select a sample of 2*kShot image patches
        sample = sample.to(device)
        # Only iterate over as many samples as there are tasks
        if idx < len(noiseTaskParams):

            # Store the noise parameters for the tak
            noiseParam = noiseTaskParams[idx]

            # Split data into train and test partitions. Each partition should have kShot images
            trainCleanImgs, testCleanImgs = torch.tensor_split(sample, 2, dim=0)

            # Verify that patch dimensions are correct
            if use_patches:
              assert trainCleanImgs.shape == (kShot, 3, imgDim, imgDim)
              assert testCleanImgs.shape == (kShot, 3, imgDim, imgDim)

            # Create synthetic noisy images
            # Apply additive Gaussian noise
            if noiseParam[0] == "G":
                sigma = noiseParam[1]
                trainNoisyImgs = trainCleanImgs + sigma * torch.randn(*trainCleanImgs.shape).cuda()
                trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                testNoisyImgs = testCleanImgs + sigma * torch.randn(*testCleanImgs.shape).cuda()
                testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])
            # Apply signal-dependent Poisson noise 
            elif noiseParam[0] == "P":
                PEAK = noiseParam[1]
                trainNoisyImgs = torch.poisson(trainCleanImgs * PEAK) / PEAK
                trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                testNoisyImgs = torch.poisson(testCleanImgs * PEAK) / PEAK
                testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])
            
            # Adapt the DnCNN model to learn to denoise the image patches for the current task
            model = learner.clone()
            model.eval()
            for innerIteration in range(numInnerIterations):
                trainError = lossFunc(model(trainNoisyImgs), trainCleanImgs)
                model.adapt(trainError)

            # Evaluate the adapted model
            predictions = learner(testNoisyImgs)
            predictions = predictions.clip(0.0, 1.0)
            testError = lossFunc(predictions, testCleanImgs).item()
            testPSNR = psnrFunc(predictions, testCleanImgs).item()

            print(f'Noise Params: {noiseParam[1]} Meta-Val Task PSNR: {testPSNR:.04f} Noisy Image PSNR: {psnrFunc(testNoisyImgs, testCleanImgs):.04f}')     

            # Visualize the one set of image patches for each task 
            if visualizeImgs:
                fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,12))
                ax[0].imshow(torch.permute(torch.squeeze(testCleanImgs[0,:,:,:]), (1,2,0)).cpu().detach().numpy())
                ax[0].set_title("Clean Image")

                ax[1].imshow(torch.permute(torch.squeeze(testNoisyImgs[0,:,:,:]), (1,2,0)).cpu().detach().numpy())
                ax[1].set_title("Noisy Image")

                ax[2].imshow(torch.permute(torch.squeeze(predictions[0,:,:,:]), (1,2,0)).cpu().detach().numpy())
                ax[2].set_title("Denoised Image") 

            # Add error and PSNR for each task
            error += testError
            PSNR += testPSNR

    # Return average error and PSNR across all tasks
    return error / len(noiseTaskParams), PSNR / len(noiseTaskParams)

'''
Meta-train function: Train's the Meta-Denoiser model to find a good initialization for the DnCNN to generalize on multiple denoising tasks

Parameters:

innerLr:            Learning rate of DnCNN

outerLr:            Learning rate of MAML

numOuterIterations: Number of outer iterations to use when updating the initialization of the DnCNN model 

numInnerIterations: Number of inner iterations to use when adapting the DnCNN model

kShot:              Number of image patches to learn on for each task

imgDim:             Spatial dimensions of the image patches 

noiseTaskParams:    List of tuples. Each tuple (p1, p2) represents the noise parameters for a task where p1 = "G" for Gaussian denoising 
                    and p1 = "P" for Poisson denoising. p2 corresponding to the noise level. For Gaussian denoising p2 = standard deviation (sigma).
                    For Poisson denoising p2 = scaling factor (beta).

metaValFreq:        How often to meta-validate the model. Every k outer iterations where k = metaValFreq. 
'''
def train(innerLr, outerLr, numOuterIterations, numInnerIterations, kShot, imgDim, noiseTaskParams, metaValFreq):

    # Verify that the model is using the GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # Instantiate the DnCNN model that MAML will run on top of
    dncnnModel = net(in_nc=3, out_nc=3, nc=64, nb=5, act_mode='BR', use_bias=True)
    dncnnModel.to(device)

    # Instantiate the MAML model for our Meta-Denoiser
    mamlModel = l2l.algorithms.MAML(dncnnModel, lr=innerLr)
    # Adam optimizer updates the DnCNN weight initialization 
    optimizer = optim.Adam(mamlModel.parameters(), lr=outerLr)

    # Define MSE loss
    mseLossFunc = nn.MSELoss()

    # Define PSNR metric
    psnrFunc = PSNR().double().cuda()

    # Keep track of PSNR and loss
    metaTrainPSNR = []
    metaTrainLoss = []
    bestAvgPSNR = 0
    dynRange = [0.0, 1.0]

    # Iterate over all tasks and update the DnCNN weight initialization at the
    # end of each outer iteration 
    for outerIteration in range(numOuterIterations):

        optimizer.zero_grad()

        # create datasets and dataloader for the meta-train data 
        train_dataset = BSDS300Dataset(patch_size=32, split='train', use_patches=True)
        # Should return data set of size (numTasks, 2*kShot, numChannels=3, height, width)
        train_dataloader = DataLoader(train_dataset, batch_size=2 * kShot, shuffle=True, drop_last=True)

        iterError = 0.0
        iterPSNR = 0.0

        # Iterate over all tasks
        iterTimeStart = time.time()
        for idx, sample in enumerate(train_dataloader):

            # Select a sample of 2*kShot image patches
            sample = sample.to(device)
            # Only iterate over as many samples as there are tasks
            if idx < len(noiseTaskParams):
                
                noiseParam = noiseTaskParams[idx]
                # Split data into train and test partitions. Each partition should have kShot images
                trainCleanImgs, testCleanImgs = torch.tensor_split(sample, 2, dim=0)

                # Verify that patch dimensions are correct
                assert trainCleanImgs.shape == (10, 3, imgDim, imgDim)
                assert testCleanImgs.shape == (10, 3, imgDim, imgDim)

                # Create synthetic noisy images
                # Apply additive Gaussian noise
                if noiseParam[0] == "G":
                    sigma = noiseParam[1]
                    trainNoisyImgs = trainCleanImgs + sigma * torch.randn(*trainCleanImgs.shape).cuda()
                    trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                    testNoisyImgs = testCleanImgs + sigma * torch.randn(*testCleanImgs.shape).cuda()
                    testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])
                # Apply signal-dependent Poisson noise
                elif noiseParam[0] == "P":
                    PEAK = noiseParam[1]
                    trainNoisyImgs = torch.poisson(trainCleanImgs * PEAK) / PEAK
                    trainNoisyImgs = trainNoisyImgs.clip(dynRange[0], dynRange[1])

                    testNoisyImgs = torch.poisson(testCleanImgs * PEAK) / PEAK
                    testNoisyImgs = testNoisyImgs.clip(dynRange[0], dynRange[1])

                # Adapt the DnCNN model to learn to denoise the image patches for the current task
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
        # Keep track of the best model (model that results in highest meta-validation PSNR)
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

'''
Meta-test function: Evaluates the model's ability to fine tune and generalize to a new unseen denoise task. 

Parameters:
numInnerIterations: Number of inner iterations to use when adapting the DnCNN model

kShot:              Number of image patches to learn on for each task

innerLr:            Learning rate of DnCNN

modelStatePath:     Name and/or path to the .pth.tar file or model weights that resulted in the best meta-validation PSNR

folderName:         Internal folder where you would like to store the denoised images 

datasetName:        Set to "SIDD" to meta-test on real noisy images. Set to "BSDS300" to meta-test on synthetic noisy images. 

noiseParam:         Noise parameter tuple. The tuple (p1, p2) represents the noise parameters for a task where p1 = "G" for Gaussian denoising 
                    and p1 = "P" for Poisson denoising. p2 corresponding to the noise level. For Gaussian denoising p2 = standard deviation (sigma).
                    For Poisson denoising p2 = scaling factor (beta).

                    Note: This parameter is only used if datasetName == "BSDS300"
'''
def test(numInnerIterations, kshot, innerLr, modelStatePath, folderName, datasetName="BSDS300", noiseParam=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dynRange = [0.0, 1.0]

    # Split data into partition for fine tuning and partition for final results
    if datasetName == "SIDD":
        test_dataset_patches = SIDDDataset(root='./SIDD_Small_sRGB_Only/Data/*/', patch_size=32, use_patches=True)
        test_dataloader_patches = DataLoader(test_dataset_patches, batch_size=kshot, shuffle=False, drop_last=True)

        test_dataset_full = SIDDDataset(root='./SIDD_Small_sRGB_Only/Data/*/', use_patches=False)
        test_dataloader_full = DataLoader(test_dataset_full, batch_size=1, shuffle=False, drop_last=True)
    else:
        test_dataset_patches = BSDS300Dataset(patch_size=32, split='test', use_patches=True)
        test_dataloader_patches = DataLoader(test_dataset_patches, batch_size=kshot, shuffle=False, drop_last=True)

        test_dataset_full = BSDS300Dataset(patch_size=32, split='test', use_patches=False)
        test_dataloader_full = DataLoader(test_dataset_full, batch_size=1, shuffle=False, drop_last=True)

    # Instantiate the DnCNN model that MAML will run on top of
    dncnnModel = net(in_nc=3, out_nc=3, nc=64, nb=5, act_mode='BR', use_bias=True).to(device)

    # Initialize the MAML model for our Meta-Denoiser and load the weights
    # saved from the best meta-validation iteration 
    mamlModel = l2l.algorithms.MAML(dncnnModel, lr=innerLr)
    mamlModel.load_state_dict(torch.load(modelStatePath))

    # Define loss
    mseLossFunc = nn.MSELoss()
    # Define PSNR metric
    psnrFunc = PSNR().double().cuda() if device == 'cuda' else PSNR().double()

    # Fine-tune the DnCNN model to learn to denoise the image patches for the current task
    learner = mamlModel.clone()
    learner.eval()
    for innerIteration in range(numInnerIterations):
        # Iterate over 1 set of kshot patches for fine tuning 
        for idx, sample in enumerate(test_dataloader_patches): 
            # Initialize data with real noisy images if using SIDD dataset
            if datasetName == "SIDD":
                sampleReal, sampleNoisy = sample
                
                sampleReal = sampleReal.to(device)
                sampleNoisy = sampleNoisy.to(device)

                trainError = mseLossFunc(learner(sampleNoisy), sampleReal)
            
            # Initialize data with synthetic noisy images if using BSDS300 dataset
            else:
                sample = sample.to(device)
                # Apply additive Gaussian noise
                if noiseParam[0] == "G":
                    sigma = noiseParam[1]
                    sampleNoisy = sample + sigma * torch.randn(*sample.shape).cuda()
                    sampleNoisy = sampleNoisy.clip(dynRange[0], dynRange[1])
                # Apply signal-dependent Poisson noise
                elif noiseParam[0] == "P":
                    PEAK = noiseParam[1]
                    sampleNoisy = torch.poisson(sample * PEAK) / PEAK
                    sampleNoisy = sampleNoisy.clip(dynRange[0], dynRange[1])

                trainError = mseLossFunc(learner(sampleNoisy), sample)

            learner.adapt(trainError)
            break

    # Evaluate the full sized test images to see if meta-denoiser model can
    # generalize to unseen tasks 
    testError = []
    testPSNR = []
    testPSNRNoisy = []
    for idx, testSample in enumerate(test_dataloader_full):
        print(idx)
        # Initialize data with real noisy images if using SIDD dataset
        if datasetName == "SIDD":
            sample, dimensions = testSample
            testReal, testNoisy = sample

            testReal = testReal.to(device)
            testNoisy = testNoisy.to(device)

        # Initialize data with synthetic noisy images if using BSDS300 dataset
        else:
            testReal = testSample.to(device)
            # Apply additive Gaussian noise
            if noiseParam[0] == "G":
                sigma = noiseParam[1]
                testNoisy = testReal + sigma * torch.randn(*testReal.shape).cuda()
                testNoisy = testNoisy.clip(dynRange[0], dynRange[1])
            # Apply signal-dependent Poisson noise
            elif noiseParam[0] == "P":
                PEAK = noiseParam[1]
                testNoisy = torch.poisson(testReal * PEAK) / PEAK
                testNoisy = testNoisy.clip(dynRange[0], dynRange[1])

        # Evaluate the adapted model
        predictions = learner(testNoisy)
        predictions = predictions.clip(0.0, 1.0)

        # Save the noisy images
        arrayNoisy = torch.permute(torch.squeeze(testNoisy), (1,2,0)).cpu().detach().numpy()
        resizedImg = arrayNoisy
        imNoisy = Image.fromarray((resizedImg * 255).astype(np.uint8))
        if datasetName == "SIDD":
            imNoisy.save("results/Image_"+str(idx)+"_Noisy.jpeg")
        else:
            imNoisy.save("BSDSResultsNew/"+folderName+"/Image_"+str(idx)+"_Noisy.jpeg")

        # Save the clean images
        arrayClean = torch.permute(torch.squeeze(testReal), (1,2,0)).cpu().detach().numpy()
        resizedImg = arrayClean
        imClean = Image.fromarray((resizedImg * 255).astype(np.uint8))
        if datasetName == "SIDD":
            imClean.save("results/Image_"+str(idx)+"_Clean.jpeg")
        else:
            imClean.save("BSDSResultsNew/"+folderName+"/Image_"+str(idx)+"_Clean.jpeg")

        # Save the denoised images
        arrayDenoised = torch.permute(torch.squeeze(predictions), (1,2,0)).cpu().detach().numpy()
        resizedImg = arrayDenoised
        imDenoised = Image.fromarray((resizedImg * 255).astype(np.uint8))
        if datasetName == "SIDD":
            imDenoised.save("results/Image_"+str(idx)+"_Denoised.jpeg")
        else:
            imDenoised.save("BSDSResultsNew/"+folderName+"/Image_"+str(idx)+"_Denoised.jpeg")

        # Save error and PSNR of each noisy and denoised image pair
        testError.append(mseLossFunc(predictions, testReal).item())
        testPSNRNoisy.append(psnrFunc(testNoisy, testReal).item())
        testPSNR.append(psnrFunc(predictions, testReal).item())

    # Return average error and PSNR over all full size images in meta-test dataset
    avgError = sum(testError) / len(testError)
    avgPSNR = sum(testPSNR) / len(testPSNR)
    avgPSNRNoisy = sum(testPSNRNoisy) / len(testPSNRNoisy)
    print(f'Meta-Test Loss: {avgError:.04f} | Meta-Test PSNR: {avgPSNR:.04f} | Average Noisy PSNR: {avgPSNRNoisy:.04f}')
