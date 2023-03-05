import copy


def train_model(model, dataloader, criteria, optimizer, scheduler, num_epochs, params):
    pretrained = params['model_files'][1]
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    tol = params['tol']
    dl = dataloader

    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl), criteria, optimizer, scheduler, pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        model = pretrained_model

    else:
        try:
            model.load_state_dict(torch.load(pretrained))
        except:
            print("pretrain error")

    kmeans(model, copy.deepcopy(dl), params)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000

    output_distrubution, preds_prev = claculate_predictions(model, copy.deepcopy(dl), params)
    target_distribution = target(output_distrubution)

    for epoch in range(num_epochs):
        scheduler.step()
        model.train()

        running_loss = 0
        running_loss_rec = 0
        running_loss_clust = 0

        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device="cuda")




