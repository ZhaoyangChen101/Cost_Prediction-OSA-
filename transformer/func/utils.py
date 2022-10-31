import torch


def position_idx(tokens: list) -> list:
    """
    Get a list of positions based on the list `tokens`.

    :param tokens: A list.
    :return: A list of index numbers of the list.
    """
    pos = []
    flag = 0

    for token in tokens:
        pos.append(flag)
        flag += 1

    return pos


def seq_padding(tokens: list, max_len: int, symbol: int) -> list:
    """
    Append the list `tokens` with the given symbol repetitively until
    the list reaches the length of max_len.

    :param tokens: A list.
    :param max_len: Max length.
    :param symbol: Padding token.
    :return: A list of max_length with elements of both `tokens` and symbol.
    """
    pad = symbol
    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if i < token_len:
            seq.append(tokens[i])
        else:
            seq.append(pad)

    return seq


def check_topk(chosen_k: int, true_label: torch.Tensor, model_output: torch.Tensor) -> float:
    """
    Calculate top-k accuracy based on true_labels and model_outputs.

    :param chosen_k: It sets the value of k in top-k accuracy.
    :param true_label: The label target with a dimension of [batch_size, sequence_length]
    :param model_output: The model output with a dimension of [batch_size, sequence_length, num_tokens]
    :return: A float number within 100.
    """

    batch_topks = []

    with torch.no_grad():
        _, topkoutputs = torch.topk(model_output, chosen_k)
        # topkoutputs: [batch_size, sequence_length, chosen_k]

        labelcount_total = 0
        topkcount = 0

        for seqidx, sequence in enumerate(true_label):
            for labelidx, truelabel in enumerate(sequence):
                if truelabel == 0:
                    break
                labelcount_total += 1
                current_topkcandidates = topkoutputs[seqidx, labelidx, :]
                if truelabel in current_topkcandidates:
                    topkcount += 1

        batch_topk = topkcount / labelcount_total
        batch_topks.append(batch_topk)

    return (sum(batch_topks) / len(batch_topks)) * 100


def cost_cal(true_labels: torch.Tensor, model_outputs: torch.Tensor, id2cost: dict, cost_tensor: torch.Tensor,
             device: torch.device) -> tuple:
    """
    Calculate annual true costs based on true_labels.
    Calculate annual expected costs based on model_outputs and cost_tensor.

    :param true_labels: The label target with a dimension of [batch_size, sequence_length]
    :param id2cost: The dictionary matching id with cost.
    :param model_outputs: The model output with a dimension of [batch_size, sequence_length, num_tokens]
    :param cost_tensor: The tensor of the cost category.
    :param device: The device to run the code.
    :return: A tuple of annual true costs and annual expected costs.
    """

    # for annual_true_costs
    annual_true_costs = []
    for sequence in true_labels:
        annual_true_cost = sum([id2cost[s.item()] for s in sequence])
        annual_true_costs.append(annual_true_cost)
    annual_true_costs = torch.Tensor(annual_true_costs).to(device)

    # for true label annual cost
    cost_tensor = cost_tensor.to(device)
    expected_costs = torch.matmul(model_outputs, cost_tensor)
    annual_expected_costs = torch.sum(expected_costs, 1)

    return annual_true_costs, annual_expected_costs
