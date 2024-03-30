import os
import time
import pandas as pd
import cv2
import numpy as np

from bye140_sdk import BackYardGripper
from gsrobotics.gelsight import gs3drecon
from gsrobotics.gelsight import gsdevice


class PIDController:
    def __init__(self, k_p: float, k_i: float, k_d: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0

    def _update_error(self, current_error: float):
        self.i_error += current_error
        self.d_error = current_error - self.p_error
        self.p_error = current_error

    def get_result(self, current_error: float, make_up_coefficient=1.0):
        self._update_error(current_error)
        return (-self.k_p * self.p_error - self.k_i * self.i_error - self.k_d * self.d_error) * make_up_coefficient

    def reset(self):
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0


def save_video(parent_path, file_name, f0):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_rgb = cv2.VideoWriter(os.path.join(parent_path,'rgb_'+file_name), fourcc, 8.5, (f0.shape[1], f0.shape[0]), isColor=True)
    print(f'Saving rgb video to {file_name}')
    out_depth = cv2.VideoWriter(os.path.join(parent_path,'depth_'+file_name), fourcc, 8.5, (f0.shape[1], f0.shape[0]), isColor=False)
    print(f'Saving depth video to {file_name}')
    return out_rgb,out_depth


def main():
    parent_folder = 'exp_data'
    folder_name = 'name'
    folder_path = os.path.join(parent_folder, folder_name)

    # 检查文件夹是否存在，如果不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    gripper = BackYardGripper()
    pid = PIDController(0.1, 0, 0.1)
    m00_log = []
    target_pos_log = []
    current_pos_los = []
    current_vel_log = []
    current_force_log = []
    time_log=[]

    target_pos = 100
    target_m00 = 10
    # Set flags
    SAVE_FLAG = True
    FIND_ROI = False
    GPU = True
    MASK_MARKERS_FLAG = True

    # Path to 3d model
    path = '.'

    dev_right = gsdevice.Camera("GelSight Mini R0B 2871-46HU")
    net_file_path = 'nnmini.pt'

    dev_right.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev_right)
    nn.load_nn(net_path, gpuorcpu)

    f0 = dev_right.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if SAVE_FLAG:
        file_path = 'log.mov'
        out_rgb,out_depth = save_video(folder_path,file_path, f0=f0)

    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('roi = ', roi)
    print('press q on image to exit')
    print('waiting gelsight depth zeroing')
    count = 0
    try:
        while dev_right.while_condition:
            start_time = time.time()
            # get the roi image
            f1 = dev_right.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG) / 30 * 255
            rounded_data = np.clip(np.round(dm), 0, 255)
            dm_uint8 = rounded_data.astype(np.uint8)
            cv2.imshow('Depth Map', dm_uint8)  # 240*320
            if count ==60:
                start_log_time=time.time()
            if count > 60:
                M = cv2.moments(dm_uint8)
                m00 = M["m00"] / 320 / 240  # 0~50
                delta_pos = pid.get_result(target_m00 - m00)
                target_pos += delta_pos
                status = gripper.get_status()
                gripper.moveto(target_pos, 150, 500, 0.2, tolerance=1, waitflag=False)
                print("m00", m00, "target_pos", target_pos, "delta_pos", delta_pos)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if SAVE_FLAG:
                    out_rgb.write(f1)
                    out_depth.write(dm_uint8)
                    m00_log.append(m00)
                    target_pos_log.append(target_pos)
                    current_pos_los.append(status["pos"])
                    current_vel_log.append(status["speed"])
                    current_force_log.append(status["force"])
                    time_log.append(time.time()-start_log_time)

            count += 1
            end_time = time.time()
            print("freq:", 1 / (end_time - start_time))

    except KeyboardInterrupt:
        print('Interrupted!')
        gripper.moveto(140, 150, 500, 0.5, tolerance=10, waitflag=False)
        df_log={'time':time_log,'m00': m00_log, 'target_pos': target_pos_log, 'current_pos': current_pos_los, 'current_vel': current_vel_log, 'current_force': current_force_log}
        df = pd.DataFrame(df_log)
        df.to_csv(os.path.join(folder_path,'adaptive_grasp_log.csv'),mode='w',index=True,header=True)
        dev_right.stop_video()


if __name__ == "__main__":
    main()
