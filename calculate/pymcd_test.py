from pymcd.mcd import Calculate_MCD

# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
mcd_toolbox = Calculate_MCD(MCD_mode="plain")
wav1 = '/home/rtrt505/speechst1/DiffVC/VCTK/wavs/p227/p227_003_mic1.wav'
wav2 = '/home/rtrt505/speechst1/DiffVC/VCTK/wavs/p238/p238_003_mic1.wav'
# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
mcd_value = mcd_toolbox.calculate_mcd(wav1, wav2)
print(mcd_value)