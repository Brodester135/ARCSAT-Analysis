import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
from astropy.stats import sigma_clip
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.centroids import centroid_com
import polars
import pathlib
import seaborn
import glob
import gc


def find_centroid_pixel(image_file, x, y):
    
    with fits.open(image_file) as hdul:
        data = hdul[0].data
    print(f"Image {image_file}: Initial x={x:.2f}, y={y:.2f}, Shape={data.shape}")
    if not (0 <= int(x)-7 < data.shape[1] and 0 <= int(y)-7 < data.shape[0]): #ensures our centroid is in bound
        print(f"Initial centroid out of bounds for {image_file}: x={x}, y={y}")
        return None
    
    cutout = data[int(y)-7:int(y)+8, int(x)-7:int(x)+8] #accounts for an area around each coordinate (15x15 cutout)

    dx, dy = centroid_com(cutout)

    del data, cutout
    gc.collect()
    
    return (x - 7 + dx, y - 7 + dy)


def measure_photometry(image_file, positions, r=9.7, sky_rin=17.9, sky_rout=22, sky_width=4.1):
    with fits.open(image_file) as hdul:
        data = hdul[0].data

    apertures = CircularAperture(positions, r=r)
    annuli = CircularAnnulus(positions, r_in=sky_rin, r_out=sky_rout)
    raw_flux = aperture_photometry(data, apertures)
    sky_flux = aperture_photometry(data, annuli)
    back = sky_flux['aperture_sum'] / annuli.area
    net_flux = raw_flux['aperture_sum'] - back * apertures.area
    
    if np.any(net_flux <= 0):
        print(f"Negative or zero net flux detectec in {image_file}: {net_flux}")
        return None, None
    
    del data, sky_flux, back
    gc.collect()

    return net_flux, raw_flux['aperture_sum']


def differential_photometry(image_list, target_pix, comp_pix, aperture=5, save_npy=False):
    target_fluxes = []
    times = []
    target_flux_nocomp = []
    comp_fluxes = [[] for _ in comp_pix]

    for img in image_list:
        print(f"\nProcessing image: {img}")
        target_xy = find_centroid_pixel(img, *target_pix)
        comp_xy = [find_centroid_pixel(img, *pix) for pix in comp_pix]

        all_xy = [target_xy] + comp_xy

        print("Target XY:", target_xy)
        print("Comparison XYs:", comp_xy)

        net_flux, _ = measure_photometry(img, all_xy, r=aperture)


        target_flux_clipped = sigma_clip([net_flux[0]], sigma=3, maxiters=3)
        if target_flux_clipped.mask[0]:
            debug_centroid(img, *target_pix, f"centroid_target{img.name}.png")
            continue
        target_flux = target_flux_clipped[0]
        comp_mean = np.mean(sigma_clip(net_flux[1:], sigma=3, maxiters=3))
        target_fluxes.append(target_flux / comp_mean)
        target_flux_nocomp.append(target_flux)
        time = Time(fits.getheader(img)['DATE-OBS']).mjd
        times.append(time)

        for i, f in enumerate(net_flux[1:]):
            comp_fluxes[i].append(f)

        del net_flux, comp_mean, target_flux_clipped, target_flux
        gc.collect()

        if save_npy:
            np.save("target_flux_nocomp.npy", target_flux_nocomp)

    return np.array(times), np.array(target_fluxes), np.array(comp_fluxes), np.array(target_flux_nocomp)


def debug_centroid(image_file, x, y, output="centroid_debug.png"):
    
    with fits.open(image_file) as hdul:
        data = hdul[0].data

    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
    plt.imshow(data, origin='lower', norm=norm)
    plt.plot(x, y, 'rx')
    cutout = data[int(y)-7:int(y)+8, int(x)-7:int(x)+8]
    dx, dy = centroid_com(cutout) if not np.all(np.isnan(cutout)) else (0,0)
    plt.plot(x - 7 + dx, y - 7 + dy, 'g+', label='Refined')
    plt.savefig(output, dpi=300)

    plt.close('all')
    del data, cutout
    gc.collect()

    print(f"Debug plot saved to {output}")


def plot_light_curves(times, diff_flux, output="lightcurve.png"):
    
    times = np.array(times)
    diff_flux = np.array(diff_flux)

    plt.figure(figsize=(8, 5))
    plt.scatter(times, diff_flux, color='Red')
    plt.xlabel("Time of Observation (MJD)")
    plt.ylabel("Differential Flux")
    plt.title("Differential Light Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    plt.close('all')
    

def plot_phase_curve(times, diff_flux, period, output="phasecurve.png"):
    
    times = np.array(times)
    diff_flux = np.array(diff_flux)

    valid = ~np.isnan(diff_flux) & (diff_flux > -1e10)
    times = times[valid]
    diff_flux = diff_flux[valid]

    times = Time(times, format='mjd').value

    # Fixed T0 from Yang (in MJD)
    T_0 = 54957.191639

    mags = -2.5 * np.log10(diff_flux)
    phase1 = ((times - T_0) / period + 0.5) % 1
    phase2 = phase1 + 1


    plt.figure(figsize=(8, 5))
    plt.scatter(phase1, mags, color='blue', label='Phase 0–1')
    plt.scatter(phase2, mags, color='red', label='Phase 1–2')
    plt.xlabel("Phase")
    plt.ylabel("Absolute Magnitude")
    plt.title("Phase-Magnitude Light Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(output, dpi=300)
    plt.close()

    plt.close('all')
    

if __name__ == "__main__":

    reduced_dir = pathlib.Path("data").resolve()
    
    #Differential Photometry values LPSEB35	240.184(deg)	+43:08(deg)
    target_pix = (505.8, 503.7)
    target_pix = (505.8 - 100, 503.7 - 100) #account for wcs cropping


    comp_pix = [(483.4, 618.8), (668.186, 204.731),(495.8, 752)]
    comp_pix = [(483.4 - 100, 618.8- 100),(668.186 - 100, 204.731 - 100),(495.8 - 100, 752 - 100)]
    #account for wcs cropping

    image_list = sorted(pathlib.Path(reduced_dir).glob('reduced_science*_reprojected.fits'))

    if not image_list:
        print(f"No reprojected images found in {reduced_dir} with pattern 'reduced_science*'")

    times, diff_flux, comp_fluxes, raw_flux = differential_photometry(image_list, target_pix, comp_pix, save_npy=True)

    plot_phase_curve(times=times, diff_flux=diff_flux, period=0.25, output="phasecurve.png")
    plot_light_curves(times=times, diff_flux=diff_flux, output="lightcurve.png")