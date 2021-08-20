from .imports import *
def analyze_model(dimension,val_dl,val_size,model,device,batch_size=32):
    """Extract Latent Space Variables, Reconcstructed Spectra and y_CN"""
    num_batches=len(val_dl) 
    reconstructed_spectra, mu, sigma = np.zeros((val_size,1,100)), np.zeros((val_size,dimension)),np.zeros((val_size,dimension))
    z = np.zeros((val_size,dimension))
    y_CN = np.zeros((val_size,2))
    model.to(device)
    for index, (xdata, ydata) in enumerate(val_dl):
        recon_, mu_,sigma_ = model(xdata.float().cuda())
        z_ = model.reparameterize(mu_,sigma_)
        
        if index == num_batches-1: ## Can be fixed...instead of dropping
            reconstructed_spectra[batch_size*index:,:,:] = recon_.cpu().detach().numpy()
            mu[batch_size*index:] = mu_.cpu().detach().numpy()
            sigma[batch_size*index:] = sigma_.cpu().detach().numpy()
            y_CN[batch_size*index:] = ydata.cpu().detach().numpy()
            z[batch_size*index:] = z_.cpu().detach().numpy()
        
        reconstructed_spectra[batch_size*index:batch_size*(index+1),:,:] = recon_.cpu().detach().numpy()
        mu[batch_size*index:batch_size*(index+1)] = mu_.cpu().detach().numpy()
        sigma[batch_size*index:batch_size*(index+1)] = sigma_.cpu().detach().numpy()
        y_CN[batch_size*index:batch_size*(index+1)] = ydata.cpu().detach().numpy()
        z[batch_size*index:batch_size*(index+1)] = z_.cpu().detach().numpy()
    
    return reconstructed_spectra, mu, sigma, z, y_CN



def vae_output(dataloader,model):
    x_one_batch, y_one_batch = next(iter(dataloader))
    print("One batch shape: ",x_one_batch.shape, y_one_batch.shape)
    print("Selecting first 4 items only...")
    y_in = y_one_batch[:4,:]
    x_in = x_one_batch[:4,:,:]
    m = model.cuda()#to(device)
    output, mu, var = m(x_in.float().cuda())
    print(len(output), mu.shape, var.shape)
    print("Output shape: ",output.shape)
    print("Latent Space Mu: ", mu[:4])
    print("Latent space Sigma: ", var[:4])
    plt.figure(figsize=(15,5))
    show_spectra_batch((x_in,y_in))
    show_spectra_batch((output.detach().cpu().numpy(), y_in))
    plt.show()


    
def plot_histograms_mu_logvar(mu, sigma):
    print("Plotting Histograms of Latent Space...\n")
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    plt.ticklabel_format(style='sci')
    ax[0].set(title ="Histograms of $\mu$ Latent Space", xlabel="$\mu$", ylabel="Counts")
    ax[1].set(title ="Histograms of $\sigma$ Latent Space", xlabel="$\sigma ( 10^{-3}$", ylabel="Counts")
    ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x*1000)))

    for index in range(mu.shape[1]):
        sns.distplot(mu[:,index],ax=ax[0])
        sns.distplot(sigma[:,index], ax=ax[1])
        
    return None

def plot_zdata(z):
    print("Plotting Scatterplot of Latent Space...\n")
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    ax[0].set(title ="Scatterplot(Only first and second axes) of Latent Space", xlabel="$z_1$", ylabel="$z_2$")
    ax[1].set(title ="Histogram of Latent Space", xlabel="$z_1$", ylabel="$z_2$")
    sns.scatterplot(z[:,0],z[:,1],ax=ax[0])
    sns.distplot(z,ax=ax[1])
    plt.show()


    
    ##export

def vae_test_output(dataloader,model, **kwargs):
    x_one_batch, y_one_batch = next(iter(dataloader))
    print("One batch shape: ",x_one_batch.shape, y_one_batch.shape)
    print("Selecting first 4 items only...")
    y_in = y_one_batch[:4,:]
    x_in = x_one_batch[:4,:,:]
    m = model.cuda()#to(device)
    output, mu = m(x_in.float().cuda())
    #print(len(output), mu.shape, var.shape)
    print("Output shape: ",output.shape)
    print("Latent Space Mu: ", mu[:4])
    #print("Latent space Sigma: ", var[:4])
    fig = plt.figure(figsize=(15,5))
    x_axis = kwargs.pop('x_axis')
    savefig_filename = kwargs.pop('savefig')
    if x_axis is not None:
        show_spectra_batch((x_in,y_in), energymesh=x_axis, plotlines=True, **kwargs)
        show_spectra_batch((output.cpu().detach().numpy(), y_in), energymesh=x_axis, **kwargs)
    else:
        show_spectra_batch((x_in,y_in), plotlines=True, **kwargs)
        show_spectra_batch((output.cpu().detach().numpy(), y_in), **kwargs)
    
    if savefig_filename:
        plt.savefig(f"{savefig_filename}.eps", dpi=150)
    plt.show()
    
    return fig
    
    
def test_output(dataloader,model, **kwargs):
    x_one_batch, y_one_batch = next(iter(dataloader))
    print("One batch shape: ",x_one_batch.shape, y_one_batch.shape)
    print("Selecting first 4 items only...")
    y_in = y_one_batch[:4,:]
    x_in = x_one_batch[:4,:,:]
    output = model(x_in.cuda().float())
    print("Output shape: ",output.shape)
    
    fig = plt.figure(figsize=(15,5))
    x_axis = kwargs.pop('x_axis')
    savefig_filename = kwargs.pop('savefig')
    if x_axis is not None:
        show_spectra_batch((x_in,y_in), energymesh=x_axis, plotlines=True, **kwargs)
        show_spectra_batch((output.cpu().detach().numpy(), y_in), energymesh=x_axis, **kwargs)
    else:
        show_spectra_batch((x_in,y_in), plotlines=True, **kwargs)
        show_spectra_batch((output.cpu().detach().numpy(), y_in), **kwargs)
    
    if savefig_filename:
        plt.savefig(f"{savefig_filename}.eps", dpi=150)
    plt.show()
    
    return fig

##export
def show_spectra_batch(sample_batched, Batch_No=None, energymesh=np.array(energymesh), **kwargs):
    Abs_batch, CN_batch = sample_batched
    
    batch_size = len(Abs_batch)

    for i in range(batch_size):
        #_ = plt.scatter(Energy_batch[i,:,:], Abs_batch[i,:,:])
        #_ = plt.title("Batch from Dataset")
        ax = plt.subplot(1, 4, i+1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i}')
        
        #ax.set_title(f'Sample #{i} of Batch No. {Batch_No}')
        
        show_data_line(energymesh, Abs_batch[i,:].reshape(-1), ax=ax, **kwargs)

##export
def show_data(energy,absorption, **kwargs):
    """ Show scatter plot with CN """
    ax = kwargs.get('ax')
    ax.set_xlabel(f"{kwargs.pop('xlabel','Energy')}")
    ax.set_ylabel(kwargs.pop('ylabel','Absorption Co-efficient'))
    #plot_kind = kwargs.pop('kind', 'scatter')
    data = pd.Series({'energy': energy, 'absorption':absorption})
    #ax = ax.map(sns.scatterplot(x='energy', y='absorption', hue='absorption', data=data, ax=ax, legend='brief', **kwargs))
    #else:
    ax = sns.scatterplot(x=energy, y=absorption, hue=absorption, legend=False, **kwargs)
    #ax.set_title(f"CN:{kwargs.get('CN')}")

def show_data_line(energy, absorption, **kwargs):
    """ Show line plot with CN """
    ax = kwargs.get('ax')
    ax.set_xlabel(f"{kwargs.pop('xlabel','Energy')}")
    ax.set_ylabel(kwargs.pop('ylabel','Absorption Co-efficient'))
    #plot_kind = kwargs.pop('kind', 'scatter')
    data = pd.Series({'energy': energy, 'absorption':absorption})
    plotlines = kwargs.pop('plotlines', False)
    if plotlines:
        ax = sns.lineplot(x='energy', y='absorption', data=data, legend=False, **kwargs)
    else:
        ax = sns.scatterplot(x='energy', y='absorption', hue='absorption', data=data, legend=False, **kwargs)
    #ax.set_title(f"CN:{kwargs.get('CN')}")
    

    
#def model_prediction_test(loss_latentSpace, test_dl, model_type, model_ckpt_path, latent_size):
#    """Plot Total and Avergae Loss separately on the test data"""
#    model = model_type.load_from_checkpoint(model_ckpt_path)
#    model.freeze() ##Important to not update the model
#    assert model.hparams.latent_size == latent_size
#    fig = test_output(test_dl, model.cuda(), x_axis=np.linspace(0.1,10,100), xlabel='Linear Space', 
#                ylabel="Normalized Values (a.u.)", savefig=False)
#               #savefig=f"{name_exp}_lt_size_{PARAMS['LatentSpaceSize']}_model_epoch_{PARAMS['epochs']}_20200409")
#    fig.show()
#
#    total, avg, std_error = calculate_test_loss(model.cuda(), test_dl, bins=20) ## For some reason the distribution of val_losses are not consistent.
#    print(total, avg, std_error)
#    loss_latentSpace.append((latent_size,(total, avg,std_error)))
#    print(loss_latentSpace)
#    return loss_latentSpace
#
#def calculate_test_loss(model, dl, bins=10):
#    """Helper function for model_prediction_test...Plot Total and Avergae Loss separately on the test data"""
#    val_losses = []
#    for batch_id, (x_data, y_data) in enumerate(dl):
#        x_data = Variable(x_data).float().cuda()
#        output = model(x_data)
#        loss = nn.MSELoss(reduction='sum')(output, x_data)
#        val_losses.append(loss)
#
#    val_losses = torch.Tensor(val_losses)#.astype('float32')
#    val_losses = val_losses.reshape(-1,1)
#
#    total, avg, std_error = torch.sum(val_losses), torch.mean(val_losses), torch.std(val_losses)/val_losses.size(0)
#    _ = plt.figure()
#    _ = plt.hist(np.array(val_losses), bins=bins)
#    _ = plt.show()
#    return (total, avg, std_error)

def model_prediction_test(loss_latentSpace, test_dl, model_type, model_ckpt_path, latent_size, loss_df):
    """Plot Total and Avergae Loss separately on the test data"""
    model = model_type.load_from_checkpoint(model_ckpt_path)
    model.freeze() ##Important to not update the model
    assert model.hparams.latent_size == latent_size
    fig = test_output(test_dl, model.cuda(), x_axis=np.linspace(0.1,10,100), xlabel='Linear Space', ylabel="Normalized Values (a.u.)", savefig=False)
               #savefig=f"{name_exp}_lt_size_{PARAMS['LatentSpaceSize']}_model_epoch_{PARAMS['epochs']}_20200409")
    fig.show()

    total, avg, std_error, losses = calculate_test_loss(model.cuda(), test_dl, bins=20) ## For some reason the distribution of val_losses are not consistent.
    print(total, avg, std_error)
    loss_latentSpace.append((latent_size,(total, avg,std_error)))
    print(loss_latentSpace)
    # val_loss_df = pd.DataFrame(np.array(losses), index=latent_size)
    loss_df.loc[latent_size-1] = [losses]
    return loss_latentSpace, loss_df

def calculate_test_loss(model, dl, bins=10):
    """Helper function for model_prediction_test...Plot Total and Avergae Loss separately on the test data"""
    val_losses = []
    for batch_id, (x_data, y_data) in enumerate(dl):
        x_data = Variable(x_data).float().cuda()
        output = model(x_data)
        loss = nn.MSELoss(reduction='sum')(output, x_data)
        val_losses.append(loss)

    val_losses = torch.Tensor(val_losses)#.astype('float32')
    val_losses = val_losses.reshape(-1,1)

    total, avg, std_error = torch.sum(val_losses), torch.median(val_losses), torch.std(val_losses)/val_losses.size(0)
    _ = plt.figure()
    _ = plt.hist(np.array(val_losses), bins=bins)
    _ = plt.show()
    return (total, avg, std_error, val_losses)