import torch
import numpy as np
from network import C3D_model
import cv2


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.float32)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = C3D_model.C3D(num_classes=101)
    checkpoint = torch.load('run/run_1/models/C3D_ucf101_epoch-39.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    video = '/Path/to/UCF-101/Biking/v_Biking_g01_c01.avi'
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if frame is None:
            continue
        tmp = center_crop(cv2.resize(frame, (171, 128)))
        tmp -= np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            clip.pop(0)
        cv2.imshow('result', frame)
        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
