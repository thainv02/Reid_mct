import torch

class Const():
    det_weight = './weights/det_10g.onnx'
    rec_weight = "./weights/w600k_r50.onnx"
    similarity_thresh = 0.47
    confidence_thresh = 0.73
    faces_dir = "./faces/"
    source = "rtsp://developer:Inf2026T1@10.29.98.57:554/cam/realmonitor?channel=1&subtype=00"
    source1 = "rtsp://developer:Inf2026T1@10.29.98.58:554/cam/realmonitor?channel=1&subtype=00"

    max_num = 0
    max_frame = 3
    #db moi
    connection_string = "mongodb://chamcong.opms.tech:27257"
    client = 'AttOBD'
    db = 'MccAttLog'