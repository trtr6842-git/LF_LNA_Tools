import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal


"""
This script takes two sets of time-domain sample data, one input and one output, and calculates the associated frequency response
It is well suited for low-frequency measurements, as multiple frequencies are analyzed at once
Any input signal can be used to excite the device-under-test (DUT), but it should contain harmonics / energy in the frequency band of interest
The transfer function is calculated ratiometrically, so the exact stimulus is not critical

A single time-domain capture is typically good for about 2 decades of calculated response with a step function input.
"""

"""
Create dummy scope data (for demo only)
"""
def dummy_data():
    ns = 10000  # number of samples
    acquisition_bits = 12  # Scope vertical SNR in effective bits
    
    noise_floor = 1 / 2**acquisition_bits  # rough approximation of quantization noise amplitude    
    tstamps = np.linspace(-0.5, 0.5, ns)  # linear ramp representing timestamps
    stimulus = np.heaviside(tstamps, 0.5)  # heaviside step function as a basic stimulus signal
    stimulus = stimulus + (np.random.rand(ns) - 0.5) * noise_floor  # simulate noise floor

    b, a = signal.butter(2, [10, 100], btype='bandpass', fs=ns)  # create a bandpass filter with a 5Hz 2nd order HP and 20Hz 2nd order LP
    response = signal.lfilter(b, a, stimulus)  # Apply the filter to the stimulus to simulate a response
    response = response  + (np.random.rand(ns) - 0.5) * noise_floor  # simulate noise floor
    
    return tstamps, stimulus, response


"""
Calculate FFT frequency response and plot results
"""
# Here is where you enter your scope data, acquiring and parsing it depends on your scope
# Make sure to apply np.assarray() to any non numpy array data types
tstamps, stimulus, response = dummy_data()  

ignore_dc = 1  # set to 0 or 1 value, 0 => includes DC offset ratio in transfer function plot, 1 => ignores DC

# Process data
dt = tstamps[1] - tstamps[0]  # get sample period
ns = len(stimulus)  # get total number of points
window = signal.windows.nuttall(ns)  # choose your window function

ws = rfftfreq(ns, dt)[ignore_dc:]  # calculate bins to frequency, ignoring DC as specified
hs = rfft(stimulus * window)[ignore_dc:]  # compute the real fft of the stimulus
hr = rfft(response * window)[ignore_dc:] # compute the real fft of the response
hs_db = 20*np.log10(np.abs(hs))  # convert to dB
hr_db = 20*np.log10(np.abs(hr))  # convert to dB

# Plot Data
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(8, 10), layout='constrained')

# Plot time domain signals
ax1.set_title('Time Domain Signals')
ax1.plot(tstamps, stimulus)  # Plot stimulus signal
ax1.plot(tstamps, response)  # Plot response signal
ax1.legend(['Stimulus', 'Response'])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Signal Level (V, A, etc)')
ax1.grid(which='both', axis='both')

# Plot FFT absolute results
ax2.set_title('FFT Results: Absolute Magnitude')
ax2.semilogx(ws, hs_db)  # Plot the stimulus FFT magnitude
ax2.semilogx(ws, hr_db)  # Plot the response FFT magnitude
ax2.legend(['Stimulus', 'Response'])
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Signal Magnitude (dB)')
ax2.grid(which='both', axis='both')

# Plot calculated frequency response
markers_on = np.arange(0, 20, 1)  # put markers on lowest few bins to show frequency resolution
ax3.set_title('FFT Results: Transfer Function Magnitude')
ax3.semilogx(ws, hr_db - hs_db, color='g', marker='o', markersize=4, markevery=markers_on)  # Plot the stimulus FFT magnitude
ax3.axhline(y = -3.0, color='r', linestyle='dashed')  # add a -3dB reference line
ax3.legend(['Magnitude', '-3dB'])
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Magnitude (dB)')
ax3.set_ylim([-60, 10])
ax3.grid(which='both', axis='both')

ax4.set_title('FFT Results: Frequency Response Phase')
ax4.semilogx(ws, 180/np.pi*np.angle(hr/hs), color='g', marker='o', markersize=4, markevery=markers_on)  # Plot the stimulus FFT magnitude
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Phase (°)')
ax4.grid(which='both', axis='both')

# fig.savefig(insert_filepath_and_name_here, dpi=300)
plt.show()