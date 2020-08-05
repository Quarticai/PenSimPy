from scipy.signal import savgol_filter
import numpy as np


def substrate_prediction(k, x, model_data):
    Raman_Spec_sg = savgol_filter(x.Raman_Spec.Intensity[k-2, :], 5, 2)
    Raman_Spec_sg_d = np.diff(Raman_Spec_sg)
    PAA_peaks_Spec = []
    PAA_peaks_Spec.extend(Raman_Spec_sg_d[349:500])
    PAA_peaks_Spec.extend(Raman_Spec_sg_d[799:860])
    x.PAA_pred.y[k - 2] = np.dot(np.array([PAA_peaks_Spec]), np.array([model_data]).T)[0]
    if k > 21:
        x.PAA_pred.y[k - 2] = (x.PAA_pred.y[k - 3] + x.PAA_pred.y[k - 4] + x.PAA_pred.y[k - 2]) / 3
    return x
