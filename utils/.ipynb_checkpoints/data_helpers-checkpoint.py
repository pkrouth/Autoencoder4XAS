from .imports import *
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import glob


def get_model_neptune(model_type, chkpoint_path=None, Neptune_Exp:int=None):
    '''
    Before using this model:
    import glob
filepath = './default/version_MSEAUT-'+str(ENTER_NEPTUNE_EXP)+'/checkpoints/*.*'
MODEL_CHECKPOINT_PATH = glob.glob(filepath)[0]
    After using this function use the following in jupyter notebook
    slider = interact(index, x=widgets.IntSlider(min=1, max=len(test_dl.dataset)-4, step=1, value=1));
    plot_specific_recons(slider.widget.result)
    '''
    if chkpoint_path:
        MODEL_CHECKPOINT_PATH = chkpoint_path
    else:
        filepath = './default/version_MSEAUT-'+str(Neptune_Exp)+'/checkpoints/*.*'
        MODEL_CHECKPOINT_PATH = glob.glob(filepath)[0]
    model = model_type.load_from_checkpoint(MODEL_CHECKPOINT_PATH).cuda()
    model.eval()
    model.freeze() ##Important to not update the model
    
    return model


def get_model_wandb(version_id=None, run_id = id, path=None, model_name = None):
    if path is not None:
        filepath = path
    else:
        filepath = f'./wandb/{run_id}/VAE/version_{version_id}/checkpoints/*.*'
    MODEL_CHECKPOINT_PATH = glob.glob(filepath)[0]
    print("...Loading Model Checkpoint from: ", MODEL_CHECKPOINT_PATH, '\n')
    #model = XasLinEncoders.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
    model = model_name.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
    #model = VaeMmdLin2XAS.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
    model.eval()
    model.freeze()
    print(model)
    
    return model

#### Old Version #### [Suitable for non VAE]
def plot_specific_reconstructions(index):
    x_input, y_input = torch.zeros((4,1,100)), torch.zeros((4,1,1)),
    for i, ind in enumerate(range(index,index+4)):
        x_input[i,:], ydata = test_dl.dataset.__getitem__(ind)
    print(x_input.shape)
    print(f"Plotting items: {index} to {index+3}")
    output = model(x_input.cuda().float())
    output = output.detach().cpu()
    fig = plt.figure(figsize=(15,5))
    show_spectra_batch((x_input,y_input), energymesh=np.linspace(0.1,10,100), plotlines=True, xlabel='Linear Space', ylabel="Normalized Values (a.u.)" )
    show_spectra_batch((output,y_input), energymesh=np.linspace(0.1,10,100), plotlines=False, xlabel='Linear Space', ylabel="Normalized Values (a.u.)" )
    fig.show()
    return index

#### New Version #### [Suitable for non VAE]
def plot_specific_reconstructions(index, test_dl, model, energymesh=None, xlabel=None):
    x_input, y_input = torch.zeros((4,1,100)), torch.zeros((4,1,1)),
    for i, ind in enumerate(range(index,index+4)):
        x_input[i,:], ydata = test_dl.dataset.__getitem__(ind)
    print(x_input.shape)
    print(f"Plotting items: {index} to {index+3}")
    if torch.cuda.is_available():
        x_input = x_input.cuda().float()
    else:
        x_input = x_input.float()
    output, mu = model(x_input)
    output = output.detach().cpu()
    fig = plt.figure(figsize=(15,5))
    if energymesh is None:
        energymesh=np.linspace(0.1,10,100)
    if xlabel is None:
        xlabel='Linear Space'
    show_spectra_batch((x_input,y_input), energymesh=energymesh , plotlines=True, xlabel=xlabel, ylabel="Normalized Values (a.u.)" )
    show_spectra_batch((output,y_input), energymesh=energymesh, plotlines=False, xlabel = xlabel, ylabel="Normalized Values (a.u.)" )
    fig.show()
    return index

def noop(x):
    return x

def encode_and_bind(original_dataframe, feature_to_encode):
    '''
    Converts a column to One-Hot-Encoded Version and append to columns
    '''
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)