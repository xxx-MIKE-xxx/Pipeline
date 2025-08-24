# pipeline_runner.py
import os, sys, json, argparse, math, glob
import numpy as np
import cv2
import onnxruntime as ort
from rtm_to_h36m import coco17_to_h36m17, normalize_xyc_frame, stitch_center_windows

def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0]-nh)//2
    left = (new_shape[1]-nw)//2
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=im.dtype)
    canvas[top:top+nh, left:left+nw] = imr
    return canvas, r, (left, top)

def yolo8_best_person(det_sess, frame):
    size = 640
    img_lb, r, (lx,ty) = letterbox(frame, (size,size))
    x = img_lb[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0..1
    x = np.transpose(x, (2,0,1))[None, ...]  # 1x3xHxW

    inputs = {det_sess.get_inputs()[0].name: x}
    outs = det_sess.run(None, inputs)
    arr = outs[0].squeeze()
    # Common exports: [84, N] or [N, 84]
    if arr.ndim == 2 and arr.shape[0] == 84:
        arr = arr.T  # [N,84]
    # columns: 0..3 = cx,cy,w,h ; class0(person) score at col 4
    if arr.shape[1] < 5:
        return None
    scores = arr[:,4]  # person class
    i = int(np.argmax(scores))
    if scores[i] < 0.25:  # low confidence
        return None
    cx, cy, w, h = arr[i,0], arr[i,1], arr[i,2], arr[i,3]
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    # map from 640 padded back to original frame
    x1 = (x1 - lx) / r; x2 = (x2 - lx) / r
    y1 = (y1 - ty) / r; y2 = (y2 - ty) / r
    # clamp and return as [x,y,w,h]
    H, W = frame.shape[:2]
    x1, y1 = float(np.clip(x1, 0, W-1)), float(np.clip(y1, 0, H-1))
    x2, y2 = float(np.clip(x2, 1, W)),   float(np.clip(y2, 1, H))
    return [x1, y1, x2-x1, y2-y1]

def crop_keep_aspect(frame, bbox, out_h=256, out_w=192):
    # Expand bbox to match out_w:out_h, then crop and resize
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    cx, cy = x + w/2.0, y + h/2.0
    target_ar = out_w / out_h
    bw, bh = w, h
    ar = bw / max(bh, 1e-6)
    if ar > target_ar:  # too wide -> grow height
        bh = bw / target_ar
    else:                # too tall -> grow width
        bw = bh * target_ar
    x1, y1 = cx - bw/2, cy - bh/2
    x2, y2 = cx + bw/2, cy + bh/2
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        crop = frame
        x1, y1 = 0, 0
    roi = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return roi, (x1, y1, (x2-x1)/out_w, (y2-y1)/out_h)  # scale back: x = x1 + nx*scale_x

def simcc_decode(simcc_x, simcc_y, split=2.0):
    # simcc_x: (1,K,W_x), simcc_y: (1,K,W_y)
    if simcc_x.ndim == 3 and simcc_x.shape[0] == 1: simcc_x = simcc_x[0]
    if simcc_y.ndim == 3 and simcc_y.shape[0] == 1: simcc_y = simcc_y[0]
    K, Wx = simcc_x.shape; Wy = simcc_y.shape[1]
    ix = np.argmax(simcc_x, axis=1)  # [K]
    iy = np.argmax(simcc_y, axis=1)  # [K]
    x = ix.astype(np.float32) / float(split)
    y = iy.astype(np.float32) / float(split)
    return np.stack([x, y], axis=1)  # [K,2] in ROI coords

def rtm_infer(sess, roi_img, roi_map, split=2.0):
    # roi_img is HxWx3 BGR uint8, resized to 256x192
    inp = roi_img[:, :, ::-1].astype(np.float32) / 255.0  # RGB
    inp = np.transpose(inp, (2,0,1))[None, ...]  # 1x3x256x192
    out = sess.run(None, {sess.get_inputs()[0].name: inp})
    # find simcc_x / simcc_y (two largest tensors)
    outs = sorted([o for o in out if isinstance(o, np.ndarray)],
                  key=lambda a: a.size, reverse=True)
    simx, simy = outs[0], outs[1]
    xy_roi = simcc_decode(simx, simy, split=split)  # [17,2]
    # back to frame coords using roi_map
    x1, y1, sx, sy = roi_map
    xy = np.zeros((17,2), dtype=np.float32)
    xy[:,0] = x1 + xy_roi[:,0] * sx
    xy[:,1] = y1 + xy_roi[:,1] * sy
    # optional confidence (if third blob present and length>=17)
    conf = np.ones((17,), np.float32)
    if len(out) >= 3 and isinstance(out[2], np.ndarray) and out[2].size >= 17:
        s = out[2].reshape(-1).astype(np.float32)
        conf = s[:17]
    xyc = np.stack([xy[:,0], xy[:,1], conf], axis=1)
    return xyc  # (17,3)

def mb_infer(sess, seq_norm_xyc):  # seq_norm_xyc: (T,17,3)
    inp = seq_norm_xyc[None, ...].astype(np.float32)  # 1xT x17x3
    out = sess.run(None, {sess.get_inputs()[0].name: inp})
    y = out[0]  # 1xT x17x3
    return y[0]

def find_rtm_end2end():
    # grab first RTMPose 'end2end.onnx' under models/**/rtmpose*/... not yolox
    cands = glob.glob("models/**/*end2end.onnx", recursive=True)
    cands = [p for p in cands if "rtmpose" in p.lower()]
    if not cands:
        raise FileNotFoundError("RTMPose end2end.onnx not found under models/**/")
    return cands[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--outdir", default="out/run1", help="Output folder")
    ap.add_argument("--rtm", default=None, help="RTMPose ONNX (defaults to auto-find)")
    ap.add_argument("--det", default="models/yolov8n.onnx", help="YOLOv8n ONNX path")
    ap.add_argument("--mbert", default="models/motionbert_lite_81.onnx", help="MotionBERT-Lite ONNX")
    ap.add_argument("--simcc_split", type=float, default=2.0, help="RTMPose SimCC split ratio (deploy.json typically 2.0)")
    ap.add_argument("--target_fps", type=int, default=20, help="FPS for MB input")
    ap.add_argument("--win", type=int, default=81, help="MB window length")
    ap.add_argument("--stride", type=int, default=27, help="MB stride")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rtm_path = args.rtm or find_rtm_end2end()
    print("RTM ONNX:", rtm_path)
    print("DET ONNX:", args.det)
    print("MB  ONNX:", args.mbert)

    providers = ["CPUExecutionProvider"]
    det_sess = ort.InferenceSession(args.det, providers=providers)
    rtm_sess = ort.InferenceSession(rtm_path, providers=providers)
    mb_sess  = ort.InferenceSession(args.mbert, providers=providers)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames2d = []  # list of (idx, 17x3)
    idx = 0
    roi = None
    while True:
        ok, frame = cap.read()
        if not ok: break
        if roi is None or (idx % max(1,int(src_fps//2)) == 0):  # redetect 2x per second
            det = yolo8_best_person(det_sess, frame)
            roi = det if det is not None else [0,0,frame.shape[1]-1, frame.shape[0]-1]
        roi_img, roi_map = crop_keep_aspect(frame, roi, out_h=256, out_w=192)
        xyc = rtm_infer(rtm_sess, roi_img, roi_map, split=args.simcc_split)  # (17,3)
        frames2d.append((idx, xyc))
        idx += 1

    cap.release()
    print(f"Captured {len(frames2d)} frames with 2D keypoints.")

    # downsample to target_fps by step in indices
    if len(frames2d) == 0:
        raise RuntimeError("No frames decoded.")
    step = max(1, int(round(src_fps / args.target_fps)))
    ds = frames2d[::step]

    # convert to H36M and normalize per-frame
    ds_h36m = []
    for _, xyc in ds:
        h36 = coco17_to_h36m17(xyc)
        h36n = normalize_xyc_frame(h36)
        ds_h36m.append(h36n)
    seq = np.stack(ds_h36m, axis=0)  # (T,17,3)
    T = seq.shape[0]

    # sliding windows into MB
    wins = []
    for s in range(0, max(0, T - args.win) + 1, args.stride):
        sub = seq[s:s+args.win]
        if sub.shape[0] < args.win:
            # pad last window by repeating last frame
            pad = np.repeat(sub[-1:], args.win - sub.shape[0], axis=0)
            sub = np.concatenate([sub, pad], axis=0)
        out3d = mb_infer(mb_sess, sub)  # (win,17,3)
        wins.append(out3d)

    merged3d = stitch_center_windows(wins, T=args.win, stride=args.stride, length=T)

    # write JSONs
    out2d = {
        "fps_src": src_fps,
        "fps_used": args.target_fps,
        "frames": [
            {"i": i, "kp2d": xyc.astype(float).tolist()}
            for (i,_), xyc in zip(ds, [f for _,f in ds])
        ]
    }
    out3d = {
        "fps": args.target_fps,
        "kp3d": merged3d.astype(float).tolist()
    }
    with open(os.path.join(args.outdir, "2d.json"), "w") as f:
        json.dump(out2d, f)
    with open(os.path.join(args.outdir, "3d.json"), "w") as f:
        json.dump(out3d, f)

    print("Wrote:", os.path.join(args.outdir, "2d.json"))
    print("Wrote:", os.path.join(args.outdir, "3d.json"))

if __name__ == "__main__":
    main()

