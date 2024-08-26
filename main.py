import os
import time
import pandas as pd
import cv2
import numpy as np

from bye140_sdk import BackYardGripper
from gsrobotics.gelsight import gs3drecon
from gsrobotics.gelsight import gsdevice

from scipy.stats import entropy

"""
光流法显示标记点的运动趋势
"""
class Flow:
    def __init__(self, col, row):
        self.x = np.zeros((row, col)).astype(int)
        self.y = np.zeros((row, col)).astype(int)
        self.col = col
        self.row = row
        self.prvs_gray = np.zeros((row, col)).astype(np.uint8)
        self.opt_flow = np.zeros((row, col, 2)).astype(np.float32)
        self.initial = 0

    def get_flow(self, img):
        if self.initial <= 20:
            self.initial += 1
            self.prvs_gray = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
        # cv2.imshow("what", self.prvs_gray)
        # cv2.waitKey(1)
        next_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.opt_flow = cv2.calcOpticalFlowFarneback(self.prvs_gray, next_gray, None, 0.5, 3, int(180 * self.col / 960),
                                                     5, 5, 1.2, 0)
        return self.opt_flow

    def get_flow_entropy(self):
        flow = self.opt_flow.reshape(-1, 2)
        histx, _ = np.histogram(flow[:, 0], bins=20, range=(-7, 7))
        histy, _ = np.histogram(flow[:, 1], bins=20, range=(-7, 7))
        probabilities_x = histx / len(flow)
        entropy_value_x = entropy(probabilities_x)
        probabilities_y = histy / len(flow)
        entropy_value_y = entropy(probabilities_y)
        return entropy_value_x, entropy_value_y

    def draw(self, img, flow, scale=5.0):  # draw the arrowedline in the image
        start_line = 40
        start_vertical = 40
        step = 20
        d_all = np.zeros((round((self.row - start_vertical) / step), round((self.col - start_line) / step), 2))
        m, n = 0, 0
        for i in range(start_vertical, self.row, step):
            for j in range(start_line, self.col, step):
                d = (flow[i, j] * scale).astype(int)
                cv2.arrowedLine(img, (j, i), (j + d[0], i + d[1]), (0, 255, 255),
                                1)  # cv2.arrowedLine(img, startpoint(x,y), endpoint(x,y), color, linedwidth)
                d_all[m, n] = d / scale
                n += 1
            m += 1
            n = 0

        return d_all

    def draw_sumline(self, img, center, sum_x, sum_y, scale=5.0):
        height = img.shape[0]
        width = img.shape[1]
        cv2.arrowedLine(img, (int(center[0]), int(center[1])),
                        (int(center[0] + sum_x * scale), int(center[1] + sum_y * scale)),
                        (0, 0, 255), 2)

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
    out_rgb_left = cv2.VideoWriter(os.path.join(parent_path, 'rgb_left_' + file_name), fourcc, 5,
                                    (f0.shape[1], f0.shape[0]), isColor=True)
    out_depth_left = cv2.VideoWriter(os.path.join(parent_path, 'depth_left_' + file_name), fourcc, 5,
                                      (f0.shape[1], f0.shape[0]), isColor=False)
    out_rgb_right = cv2.VideoWriter(os.path.join(parent_path,'rgb_right_'+file_name), fourcc, 5, (f0.shape[1], f0.shape[0]), isColor=True)
    print(f'Saving rgb video to {file_name}')
    out_depth_right = cv2.VideoWriter(os.path.join(parent_path,'depth_right_'+file_name), fourcc, 5, (f0.shape[1], f0.shape[0]), isColor=False)
    print(f'Saving depth video to {file_name}')
    return out_rgb_left,out_depth_left,out_rgb_right,out_depth_right


def main():
    parent_folder = 'exp_data'
    folder_name = 'screw'
    folder_path = os.path.join(parent_folder, folder_name)

    # 检查文件夹是否存在，如果不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    gripper = BackYardGripper()
    pid = PIDController(0.1, 0, 0.1)
    m00_log = []
    m00_left_log = []
    entropy_x_left_log = []
    entropy_y_left_log = []
    entropy_x_right_log = []
    entropy_y_right_log = []
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
    dev_left = gsdevice.Camera("GelSight Mini R0B 28GH-EJZ8")
    net_file_path = 'nnmini.pt'

    dev_right.connect()
    dev_left.connect()

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
    flow_right=Flow(f0.shape[1],f0.shape[0])
    flow_right.get_flow(f0)

    f0_left = dev_left.get_raw_image()
    roi_left = (0, 0, f0_left.shape[1], f0_left.shape[0])
    flow_left = Flow(f0_left.shape[1], f0_left.shape[0])
    flow_left.get_flow(f0_left)
    if SAVE_FLAG:
        file_path = 'log.mov'
        out_rgb_left,out_depth_left,out_rgb_right,out_depth_right = save_video(folder_path,file_path, f0=f0)

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
            bigframe_right = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            cv2.imshow('Image_right', bigframe_right)
            opti_flow_right=flow_right.get_flow(f1)

            f2 = dev_left.get_image()
            bigframe_left = cv2.resize(f2, (f2.shape[1] * 2, f2.shape[0] * 2))
            cv2.imshow('Image_left', bigframe_left)
            opti_flow_left = flow_left.get_flow(f2)

            # compute the depth map
            dm_right = nn.get_depthmap(f1, MASK_MARKERS_FLAG) / 30 * 255
            dm_left = nn.get_depthmap(f2, MASK_MARKERS_FLAG) / 30 * 255
            rounded_data = np.clip(np.round(dm_right), 0, 255)
            dm_uint8_right = rounded_data.astype(np.uint8)
            rounded_data = np.clip(np.round(dm_left), 0, 255)
            dm_uint8_left = rounded_data.astype(np.uint8)
            cv2.imshow('Depth Map Right', dm_uint8_right)  # 240*320
            cv2.imshow('Depth Map Left', dm_uint8_left)  # 240*320
            if count ==60:
                start_log_time=time.time()
            if count > 60:
                M = cv2.moments(dm_uint8_right)
                m00_right = M["m00"] / 320 / 240  # 0~50
                M_left=cv2.moments(dm_uint8_left)
                m00_left=M_left["m00"]/320/240
                entropy_x_right, entropy_y_right = flow_right.get_flow_entropy()
                entropy_x_left, entropy_y_left = flow_left.get_flow_entropy()
                delta_pos = pid.get_result(target_m00 - m00_right)
                target_pos += delta_pos
                status = gripper.get_status()
                gripper.moveto(target_pos, 150, 500, 0.2, tolerance=1, waitflag=False)
                print("m00", m00_right, "ent_x_r", entropy_x_right, "ent_y_r", entropy_y_right, 'target_pos', target_pos)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if SAVE_FLAG:
                    out_rgb_right.write(f1)
                    out_depth_right.write(dm_uint8_right)
                    out_rgb_left.write(f2)
                    out_depth_left.write(dm_uint8_left)
                    m00_log.append(m00_right)
                    m00_left_log.append(m00_left)
                    entropy_x_right_log.append(entropy_x_right)
                    entropy_y_right_log.append(entropy_y_right)
                    entropy_x_left_log.append(entropy_x_left)
                    entropy_y_left_log.append(entropy_y_left)
                    target_pos_log.append(target_pos)
                    current_pos_los.append(status["pos"])
                    current_vel_log.append(status["speed"])
                    time_log.append(time.time()-start_log_time)

            count += 1
            end_time = time.time()
            print("freq:", 1 / (end_time - start_time))

    except KeyboardInterrupt:
        print('Interrupted!')
        gripper.moveto(140, 150, 500, 0.5, tolerance=10, waitflag=False)
        df_log={'time':time_log,'m00_right': m00_log, 'm00_left':m00_left_log, 'entropy_x_right': entropy_x_right_log, 'entropy_y_right': entropy_y_right_log,
                'entropy_x_left': entropy_x_left_log, 'entropy_y_left': entropy_y_left_log, 'gripper_target_pos': target_pos,
                'gripper_pos': current_pos_los, 'gripper_vel': current_vel_log}
        df = pd.DataFrame(df_log)
        df.to_csv(os.path.join(folder_path,'adaptive_grasp_log.csv'),mode='w',index=True,header=True)
        dev_right.stop_video()


if __name__ == "__main__":
    main()
