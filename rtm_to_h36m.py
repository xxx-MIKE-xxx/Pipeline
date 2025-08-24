# rtm_to_h36m.py
import math
import numpy as np

def midpoint(a, b):
    return np.array([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0], dtype=np.float32)

def coco17_to_h36m17(coco_xyc):
    """
    coco_xyc: (17,3) float32, order:
      0 nose,1 lEye,2 rEye,3 lEar,4 rEar,5 lShoulder,6 rShoulder,7 lElbow,8 rElbow,
      9 lWrist,10 rWrist,11 lHip,12 rHip,13 lKnee,14 rKnee,15 lAnkle,16 rAnkle
    returns (17,3) H36M-ish:
      0 Pelvis,1 RHip,2 RKnee,3 RAnkle,4 LHip,5 LKnee,6 LAnkle,
      7 Spine,8 Thorax,9 Nose,10 LShoulder,11 LElbow,12 LWrist,
      13 RShoulder,14 RElbow,15 RWrist,16 HeadTop(≈Nose)
    """
    out = np.zeros((17,3), dtype=np.float32)
    # helpful midpoints
    pelvis_xy = midpoint(coco_xyc[11,:2], coco_xyc[12,:2])
    thorax_xy = midpoint(coco_xyc[5,:2],  coco_xyc[6,:2])
    spine_xy  = midpoint(pelvis_xy, thorax_xy)

    # confidences: use min of parents for synthetic joints
    pelvis_c = min(coco_xyc[11,2], coco_xyc[12,2])
    thorax_c = min(coco_xyc[5,2],  coco_xyc[6,2])
    spine_c  = min(pelvis_c, thorax_c)

    def setj(i, xy, c):
        out[i,0], out[i,1], out[i,2] = xy[0], xy[1], c

    setj(0, pelvis_xy, pelvis_c)
    setj(1, coco_xyc[12,:2], coco_xyc[12,2])  # RHip
    setj(2, coco_xyc[14,:2], coco_xyc[14,2])  # RKnee
    setj(3, coco_xyc[16,:2], coco_xyc[16,2])  # RAnkle
    setj(4, coco_xyc[11,:2], coco_xyc[11,2])  # LHip
    setj(5, coco_xyc[13,:2], coco_xyc[13,2])  # LKnee
    setj(6, coco_xyc[15,:2], coco_xyc[15,2])  # LAnkle
    setj(7, spine_xy,  spine_c)               # Spine
    setj(8, thorax_xy, thorax_c)              # Thorax
    setj(9, coco_xyc[0,:2],  coco_xyc[0,2])   # Nose
    setj(10, coco_xyc[5,:2], coco_xyc[5,2])   # LShoulder
    setj(11, coco_xyc[7,:2], coco_xyc[7,2])   # LElbow
    setj(12, coco_xyc[9,:2], coco_xyc[9,2])   # LWrist
    setj(13, coco_xyc[6,:2], coco_xyc[6,2])   # RShoulder
    setj(14, coco_xyc[8,:2], coco_xyc[8,2])   # RElbow
    setj(15, coco_xyc[10,:2],coco_xyc[10,2])  # RWrist
    setj(16, coco_xyc[0,:2], coco_xyc[0,2])   # HeadTop ≈ Nose

    return out

def normalize_xyc_frame(xyc17):
    """Normalize (x,y,conf) to 0..1 per-frame using tight bbox over xy."""
    x = xyc17[:,0]; y = xyc17[:,1]
    minx, maxx = float(x.min()), float(x.max())
    miny, maxy = float(y.min()), float(y.max())
    w = max(maxx-minx, 1e-3); h = max(maxy-miny, 1e-3)
    out = xyc17.copy()
    out[:,0] = (x - minx) / w
    out[:,1] = (y - miny) / h
    return out

def stitch_center_windows(wins3d, T=81, stride=27, length=None):
    """
    wins3d: list of (T,17,3). We keep center [27:54) from each window.
    Returns (length,17,3) with gaps filled from first/last windows.
    """
    import numpy as np
    if not wins3d: return np.zeros((0,17,3), np.float32)
    if length is None:
        # approximate total length from last window start
        length = stride*(len(wins3d)-1) + T
    out = np.full((length,17,3), np.nan, np.float32)
    for wi, w in enumerate(wins3d):
        start = wi*stride
        cs, ce = 27, 54  # center half minus edges
        dst_s = start + cs
        dst_e = start + ce
        out[dst_s:dst_e] = w[cs:ce]
    # fill leading/trailing NaNs from nearest non-NaN
    # simple forward/backward fill:
    for i in range(length):
        if np.isnan(out[i,0,0]):
            # forward search
            j = i+1
            while j<length and np.isnan(out[j,0,0]): j+=1
            if j<length: out[i] = out[j]
    for i in range(length-1, -1, -1):
        if np.isnan(out[i,0,0]):
            j = i-1
            while j>=0 and np.isnan(out[j,0,0]): j-=1
            if j>=0: out[i] = out[j]
    return out
