import torch
import torch.nn.functional as F
from func.utils import check_topk, cost_cal
import numpy as np
import sklearn.metrics as skm
import time
import copy


def train_epoch(train_DataLoader, model, optimizer, loss_function, id2cost, cost_tensor, params,
                device) -> list:
    """
    Train the train_DataLoader for one epoch.

    :param train_DataLoader: DataLoader for training.
    :param model: The initiated model to be trained.
    :param optimizer: The algorithm to optimize model parameters.
    :param loss_function: The loss function to conduct back propagation.
    :param device: The device to run the code.
    :param id2cost: The dictionary matching id with cost.
    :param cost_tensor: The tensor of the cost category.
    :param params: The dictionary storing adjustable parameters for models and functions.
    :return: a list of training results including: loss, top-k accuracy (k=3,5,10), MAE, MSE, RMSE, and R2 score.
    """

    # set the model into the training mode
    model.train()

    # lists to store training results of each batch
    batch_loss = []
    batch_top3 = []
    batch_top5 = []
    batch_top10 = []
    batch_annual_true_costs = []
    batch_annual_expected_costs = []
    # list of results for one entire epoch
    results = []

    # batch counter
    batch_counter = 0

    for inputs, input_tgt, tgt in train_DataLoader:
        # batch counter + 1
        batch_counter += 1

        # Get look-ahead mask for input target
        input_target = input_tgt[0]
        sequence_length = input_target[0].size(0)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Get padding mask for inputs and input target
        src_pad_mask = model.create_pad_mask(matrix=inputs[0], pad_token=0).to(device)
        tgt_pad_mask = model.create_pad_mask(matrix=input_target, pad_token=0).to(device)

        # Sets gradients of all model parameters to zero
        optimizer.zero_grad()

        # model execution
        output = model(inputs, input_tgt, device=device, tgt_mask=tgt_mask,
                       src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        # output: [seq_len, batch_size, num_tokens]
        # move output and tgt to device
        output = output.to(device)
        tgt = tgt.to(device)

        # calculate loss
        loss_train = loss_function(output, tgt, id2cost, cost_tensor, params, device)
        loss_train = loss_train.to(device)

        # back propagation
        loss_train.backward()
        # gradient descent
        optimizer.step()

        # store loss
        batch_loss.append(loss_train.item())

        # calculate training top-k accuracy:
        output = F.softmax(output, dim=2)
        output = output.permute(1, 0, 2)
        # output: [batch_size, seq_len, num_tokens]
        # top-3
        current_top3 = check_topk(3, tgt, output)
        batch_top3.append(current_top3)
        # top-5
        current_top5 = check_topk(5, tgt, output)
        batch_top5.append(current_top5)
        # top-10
        current_top10 = check_topk(10, tgt, output)
        batch_top10.append(current_top10)

        # calculate annual true costs and expected costs
        annual_true_costs, annual_expected_costs = cost_cal(tgt, output, id2cost, cost_tensor, device)
        # add to cost lists
        # torch.Tensor.tolist can eliminate device info
        annual_true_costs_list = torch.Tensor.tolist(annual_true_costs)
        annual_expected_costs_list = torch.Tensor.tolist(annual_expected_costs)
        batch_annual_true_costs += annual_true_costs_list
        batch_annual_expected_costs += annual_expected_costs_list

        # calculate prediction accuracy metrics for the batch
        mae = skm.mean_absolute_error(annual_true_costs_list,
                                      annual_expected_costs_list)  # error: because expected cost is nan
        mse = skm.mean_squared_error(annual_true_costs_list, annual_expected_costs_list)
        rmse = np.sqrt(mse)  # or mse**(0.5)
        r2 = skm.r2_score(annual_true_costs_list, annual_expected_costs_list)

        # print results during training
        print(f"train batch: {batch_counter}/{len(train_DataLoader)}, train loss: {loss_train.item():.3f}")
        print(f"top3 acc: {current_top3:.2f}%, top5 acc: {current_top5:.2f}%, top10 acc: {current_top10:.2f}%")
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R-Squared:", r2)


    # calculate average loss and top-k accuracy
    epoch_loss = np.mean(batch_loss)
    epoch_top3 = np.mean(batch_top3)
    epoch_top5 = np.mean(batch_top5)
    epoch_top10 = np.mean(batch_top10)
    # calculate MAE, MSE, RMSE, and R2 score for one epoch
    epoch_mae = skm.mean_absolute_error(batch_annual_true_costs, batch_annual_expected_costs)
    epoch_mse = skm.mean_squared_error(batch_annual_true_costs, batch_annual_expected_costs)
    epoch_rmse = np.sqrt(epoch_mse)
    epoch_r2 = skm.r2_score(batch_annual_true_costs, batch_annual_expected_costs)
    # add the results to list
    results.append(epoch_loss)
    results.append(epoch_top3)
    results.append(epoch_top5)
    results.append(epoch_top10)
    results.append(epoch_mae)
    results.append(epoch_mse)
    results.append(epoch_rmse)
    results.append(epoch_r2)

    return results


def eval_epoch(eval_DataLoader, model, loss_function, id2cost, cost_tensor, params, device) -> list:
    """
    Evaluate the eval_DataLoader for one epoch.
    This function applies to both validation and test data.

    :param eval_DataLoader: DataLoader for evaluation.
    :param model: The trained model to be evaluated.
    :param loss_function: The loss function to evaluate how well the model has been trained.
    :param params: The dictionary storing adjustable parameters for models and functions.
    :param id2cost: The dictionary matching id with cost.
    :param cost_tensor: The tensor of the cost category.
    :param device: The device to run the code.
    :return: a list of evaluation results including: loss, top-k accuracy (k=3,5,10), MAE, MSE, RMSE,
        and R2 score.
    """

    # set the model into the evaluation mode
    model.eval()

    # lists to store training results of each batch
    batch_loss = []
    batch_top3 = []
    batch_top5 = []
    batch_top10 = []
    batch_annual_true_costs = []
    batch_annual_expected_costs = []

    # list of results for one entire epoch
    results = []

    # batch counter
    batch_counter = 0

    with torch.no_grad():
        for inputs, input_tgt, tgt in eval_DataLoader:
            # batch counter + 1
            batch_counter += 1

            # Get look-ahead mask for input target
            input_target = input_tgt[0]
            sequence_length = input_target[0].size(0)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Get padding mask for inputs and input target
            src_pad_mask = model.create_pad_mask(matrix=inputs[0], pad_token=0).to(device)
            tgt_pad_mask = model.create_pad_mask(matrix=input_target, pad_token=0).to(device)

            # model execution
            output = model(inputs, input_tgt, device=device, tgt_mask=tgt_mask,
                           src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
            # output: [seq_len, batch_size, num_tokens]
            # move output and tgt to device
            output = output.to(device)
            tgt = tgt.to(device)

            # calculate loss
            loss_eval = loss_function(output, tgt, id2cost, cost_tensor, params, device)
            loss_eval = loss_eval.to(device)

            # store loss
            batch_loss.append(loss_eval.item())

            # calculate training top-k accuracy:
            output = F.softmax(output, dim=2)
            output = output.permute(1, 0, 2)
            # output: [batch_size, seq_len, num_tokens]
            # top-3
            current_top3 = check_topk(3, tgt, output)
            batch_top3.append(current_top3)
            # top-5
            current_top5 = check_topk(5, tgt, output)
            batch_top5.append(current_top5)
            # top-10
            current_top10 = check_topk(10, tgt, output)
            batch_top10.append(current_top10)

            # calculate annual true costs and expected costs
            annual_true_costs, annual_expected_costs = cost_cal(tgt, output, id2cost, cost_tensor, device)

            # add to cost lists
            # torch.Tensor.tolist can eliminate device info
            annual_true_costs_list = torch.Tensor.tolist(annual_true_costs)
            annual_expected_costs_list = torch.Tensor.tolist(annual_expected_costs)
            batch_annual_true_costs += annual_true_costs_list
            batch_annual_expected_costs += annual_expected_costs_list

            # calculate prediction accuracy metrics
            mae = skm.mean_absolute_error(annual_true_costs_list,
                                          annual_expected_costs_list)  # error: because expected cost is nan
            mse = skm.mean_squared_error(annual_true_costs_list, annual_expected_costs_list)
            rmse = np.sqrt(mse)  # or mse**(0.5)
            r2 = skm.r2_score(annual_true_costs_list, annual_expected_costs_list)

            # print results during validation
            print(f"eval batch: {batch_counter}/{len(eval_DataLoader)}, eval loss: {loss_eval.item():.3f}")
            print(f"top3 acc: {current_top3:.2f}%, top5 acc: {current_top5:.2f}%, top10 acc: {current_top10:.2f}%")
            print("MAE:", mae)
            print("MSE:", mse)
            print("RMSE:", rmse)
            print("R-Squared:", r2)


    # calculate average loss and top-k accuracy
    epoch_loss = np.mean(batch_loss)
    epoch_top3 = np.mean(batch_top3)
    epoch_top5 = np.mean(batch_top5)
    epoch_top10 = np.mean(batch_top10)
    # calculate MAE, MSE, RMSE, and R2 score for one epoch
    epoch_mae = skm.mean_absolute_error(batch_annual_true_costs, batch_annual_expected_costs)
    epoch_mse = skm.mean_squared_error(batch_annual_true_costs, batch_annual_expected_costs)
    epoch_rmse = np.sqrt(epoch_mse)
    epoch_r2 = skm.r2_score(batch_annual_true_costs, batch_annual_expected_costs)
    # add the results to list
    results.append(epoch_loss)
    results.append(epoch_top3)
    results.append(epoch_top5)
    results.append(epoch_top10)
    results.append(epoch_mae)
    results.append(epoch_mse)
    results.append(epoch_rmse)
    results.append(epoch_r2)

    return results


def fit(train_DataLoader, val_DataLoader, model, optimizer, loss_function, id2cost, cost_tensor, params, device,
        epochs) -> list:
    """
    Execute training and validation for a certain number of epochs.

    :param train_DataLoader: DataLoader for training.
    :param val_DataLoader: DataLoader for validation.
    :param model: The trained model to be evaluated.
    :param optimizer: The algorithm to optimize model parameters.
    :param loss_function: The loss function to evaluate how well the model has been trained.
    :param id2cost: The dictionary matching id with cost.
    :param cost_tensor: The tensor of the cost category.
    :param params: The dictionary storing adjustable parameters for models and functions.
    :param device: The device to run the code.
    :param epochs: The number of epochs for training and validation.
    :return: a tuple including 1) epoch results of train, 2) epoch results of val, and 3) best model info.
    """

    # lists to store results for training and validation data
    # train
    train_losses = []
    train_top3 = []
    train_top5 = []
    train_top10 = []
    train_mae = []
    train_mse = []
    train_rmse = []
    train_r2 = []
    # val
    val_losses = []
    val_top3 = []
    val_top5 = []
    val_top10 = []
    val_mae = []
    val_mse = []
    val_rmse = []
    val_r2 = []
    # two lists to summarize the lists above for training and validation
    train_summary = []
    val_summary = []

    # choose best_model based on the smallest validation loss
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0

    # starting time for training and validation
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        # starting time for the epoch
        epoch_start = time.time()
        print("-" * 45, f"Epoch {epoch}", "-" * 45)

        # train
        train_results = train_epoch(train_DataLoader, model, optimizer, loss_function, id2cost, cost_tensor, params,
                                    device)
        # epoch results
        epoch_loss_train = train_results[0]
        epoch_top3_train = train_results[1]
        epoch_top5_train = train_results[2]
        epoch_top10_train = train_results[3]
        epoch_mae_train = train_results[4]
        epoch_mse_train = train_results[5]
        epoch_rmse_train = train_results[6]
        epoch_r2_train = train_results[7]
        # add to list
        train_losses.append(epoch_loss_train)
        train_top3.append(epoch_top3_train)
        train_top5.append(epoch_top5_train)
        train_top10.append(epoch_top10_train)
        train_mae.append(epoch_mae_train)
        train_mse.append(epoch_mse_train)
        train_rmse.append(epoch_rmse_train)
        train_r2.append(epoch_r2_train)

        # validation
        val_results = eval_epoch(val_DataLoader, model, loss_function, id2cost, cost_tensor, params, device)
        # epoch results
        epoch_loss_val = val_results[0]
        epoch_top3_val = val_results[1]
        epoch_top5_val = val_results[2]
        epoch_top10_val = val_results[3]
        epoch_mae_val = val_results[4]
        epoch_mse_val = val_results[5]
        epoch_rmse_val = val_results[6]
        epoch_r2_val = val_results[7]
        # add to list
        val_losses.append(epoch_loss_val)
        val_top3.append(epoch_top3_val)
        val_top5.append(epoch_top5_val)
        val_top10.append(epoch_top10_val)
        val_mae.append(epoch_mae_val)
        val_mse.append(epoch_mse_val)
        val_rmse.append(epoch_rmse_val)
        val_r2.append(epoch_r2_val)

        # calculate time spent for the epoch
        epoch_time = time.time() - epoch_start
        # print results for the epoch
        print(f"Epoch {epoch} summary:\
        \n\ttrain -> avg loss: {epoch_loss_train:.3f}\
        \n\t         MAE:{epoch_mae_train:.3f}, MSE:{epoch_mse_train:.3f}, RMSE:{epoch_rmse_train:.3f}, R2: {epoch_r2_train:.3f} \
        \n\t         top3 acc: {epoch_top3_train:.2f}%, top5 acc: {epoch_top5_train:.2f}%, top10 acc: {epoch_top10_train:.2f}%\
        \n\tval   -> avg loss: {epoch_loss_val:.3f}\
        \n\t         MAE:{epoch_mae_val:.3f}, MSE:{epoch_mse_val:.3f}, RMSE:{epoch_rmse_val:.3f}, R2: {epoch_r2_val:.3f} \
        \n\t         top3 acc: {epoch_top3_val:.2f}%, top5 acc: {epoch_top5_val:.2f}%, top10 acc: {epoch_top10_val:.2f}%\
        \n\ttime  -> {epoch_time}s")

        # save the best model with the smallest validation loss
        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            best_epoch = epoch
            best_model = copy.deepcopy(model)


    # print summary
    print(f'The best_model with least val loss is in epoch {best_epoch}:')
    index = best_epoch - 1
    print(f'avg total loss: {val_losses[index]}')
    print(f'MAE:{val_mae[index]}, MSE:{val_mse[index]}, RMSE:{val_rmse[index]}, R-Squared:{val_r2[index]}')
    print(f'top3:{val_top3[index]}, top5:{val_top5[index]}, top10:{val_top10[index]}')
    print(best_model)
    # total time spent
    total_time = time.time() - total_start
    converted_format = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f'total time: {converted_format}')

    # add lists to summary lists
    # train
    train_summary.append(train_losses)
    train_summary.append(train_top3)
    train_summary.append(train_top5)
    train_summary.append(train_top10)
    train_summary.append(train_mae)
    train_summary.append(train_mse)
    train_summary.append(train_rmse)
    train_summary.append(train_r2)
    # val
    val_summary.append(val_losses)
    val_summary.append(val_top3)
    val_summary.append(val_top5)
    val_summary.append(val_top10)
    val_summary.append(val_mae)
    val_summary.append(val_mse)
    val_summary.append(val_rmse)
    val_summary.append(val_r2)

    return train_summary, val_summary, best_model
