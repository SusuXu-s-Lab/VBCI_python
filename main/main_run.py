# /main/main_run.py

import numpy as np
import rasterio
from methods import SVI, pruning
from evaluation import rocdetpr
import os

def main():
    # Initialize
    # Define data and output locations
    current_dir = os.getcwd()
    data_location = os.path.join(current_dir, 'data')
    output_location = os.path.join(current_dir, 'outputs')
    
    # Ensure output directory exists
    os.makedirs(output_location, exist_ok=True)

    # Import data
    with rasterio.open(os.path.join(data_location, 'PR_DPM.tif')) as src:
        Y = src.read(1)
        Y_R = src.transform

    with rasterio.open(os.path.join(data_location, 'PR_BD.tif')) as src:
        BD = src.read(1)
        BD_R = src.transform

    with rasterio.open(os.path.join(data_location, 'PR_LS.tif')) as src:
        LS = src.read(1)
        LS_R = src.transform

    with rasterio.open(os.path.join(data_location, 'PR_LF.tif')) as src:
        LF = src.read(1)
        LF_R = src.transform

    # Fix Data Input
    BD[BD > 0] = 1
    Y[np.isnan(Y)] = 0
    BD[np.isnan(BD)] = 0
    LS[np.isnan(LS)] = 0
    LF[np.isnan(LF)] = 0

    # Convert Landslide Areal Percentages to Probabilities
    new_LS = LS.copy()
    index = np.where(LS > 0)
    for i in zip(*index):
        p = [4.035, -3.042, 5.237, (-7.592 - np.log(LS[i]))]
        roots = np.roots(p)
        real_roots = np.real(roots[np.isreal(roots)])
        new_LS[i] = real_roots[0] if len(real_roots) > 0 else 0

    # Convert Liquefaction Areal Percentages to Probabilities
    new_LF = LF.copy()
    index = np.where(LF > 0)
    for i in zip(*index):
        try:
            new_LF[i] = (np.log((np.sqrt(0.4915 / LF[i]) - 1) / 42.40)) / (-9.165)
        except:
            new_LF[i] = 0  # Handle potential math errors

    # Change into Non-negative Probabilities
    new_LF[new_LF < 0] = 0
    new_LS[new_LS < 0] = 0
    new_LS[np.isnan(new_LS)] = 0
    new_LF[np.isnan(new_LF)] = 0
    tmp_LF = new_LF.copy()
    tmp_LS = new_LS.copy()

    # Classify Local Model by Pruning
    prune_type = 'double'
    sigma = np.median(np.abs(new_LS[(LS > 0) & (LF > 0)] - new_LF[(LS > 0) & (LF > 0)]))
    LOCAL = pruning(BD, tmp_LS, tmp_LF, sigma, prune_type)
    tmp_LS[(LOCAL == 5) | (LOCAL == 6)] = np.min(new_LS[new_LS > 0]) if np.any(new_LS > 0) else 0
    tmp_LF[(LOCAL == 5) | (LOCAL == 6)] = np.min(new_LF[new_LF > 0]) if np.any(new_LF > 0) else 0

    # Set Lambda Term
    lambda_ = 0

    # Initialize Weight Vector w
    w = np.random.rand(15)
    w[[3, 4]] = 0
    w[[0, 2]] = -1 * w[[0, 2]]
    regu_type = 1

    # Set Variational hyperparameters
    Nq = 10  # Number of Posterior Probability Iterations

    # Set Weight Updating Parameters
    rho = 1e-3  # Step size
    delta = 1e-3  # Acceptable tolerance for weight optimization
    eps_0 = 0.001  # Lower-bound non-negative weight

    # Output
    opt_w, opt_QBD, opt_QLS, opt_QLF, QLS, QLF, QBD, final_loss, best_loss, local = SVI(
        Y, tmp_LS, tmp_LF, w, Nq, rho, delta, eps_0, LOCAL, lambda_, regu_type, sigma, prune_type)

    # Convert Probabilities to Areal Percentages
    final_QLS = np.exp(-7.592 + 5.237 * opt_QLS - 3.042 * opt_QLS ** 2 + 4.035 * opt_QLS ** 3)
    final_QLF = 0.4915 / (1 + 42.40 * np.exp(-9.165 * opt_QLF)) ** 2

    # Round down very small areal percentages to zero
    final_QLS[final_QLS <= np.exp(-7.592)] = 0
    final_QLF[final_QLF <= 0.4915 / (1 + 42.40) ** 2] = 0

    # Remove probabilities in water bodies
    final_QLS[(LS == 0) & (LF == 0)] = 0
    final_QLF[(LS == 0) & (LF == 0)] = 0

    # Export GeoTIFF Files
    with rasterio.open(os.path.join(output_location, 'QLS.tif'), 'w', driver='GTiff',
                       height=final_QLS.shape[0],
                       width=final_QLS.shape[1], count=1, dtype=final_QLS.dtype,
                       crs=LS_R.crs, transform=LS_R.transform) as dst:
        dst.write(final_QLS, 1)

    with rasterio.open(os.path.join(output_location, 'QLF.tif'), 'w', driver='GTiff',
                       height=final_QLF.shape[0],
                       width=final_QLF.shape[1], count=1, dtype=final_QLF.dtype,
                       crs=LS_R.crs, transform=LS_R.transform) as dst:
        dst.write(final_QLF, 1)

    with rasterio.open(os.path.join(output_location, 'QBD.tif'), 'w', driver='GTiff',
                       height=QBD.shape[0],
                       width=QBD.shape[1], count=1, dtype=QBD.dtype,
                       crs=LS_R.crs, transform=LS_R.transform) as dst:
        dst.write(QBD, 1)

    # Save all to a file
    np.savez(os.path.join(output_location, f'lambda{lambda_}_sigma{sigma}_prune{prune_type}.npz'),
             opt_w=opt_w,
             opt_QBD=opt_QBD, opt_QLS=opt_QLS, opt_QLF=opt_QLF,
             QLS=QLS, QLF=QLF, QBD=QBD,
             final_loss=final_loss, best_loss=best_loss, LOCAL=LOCAL)

if __name__ == "__main__":
    main()
