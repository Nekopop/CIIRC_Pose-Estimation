import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
import rosbag2_py
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class PointCloudGenerator:
    def __init__(self, K, width, height):
        self.K = K
        self.u, self.v = self.precompute_grid(width, height)

    def precompute_grid(self, width, height):
        u = np.tile(np.arange(width), (height, 1))
        v = np.tile(np.arange(height)[:, np.newaxis], (1, width))
        return u, v

    def depth2points(self, depth, roi, max_points=2000):
        u_sel = self.u[roi[1]:roi[3], roi[0]:roi[2]].flatten()
        v_sel = self.v[roi[1]:roi[3], roi[0]:roi[2]].flatten()
        z_sel = depth[roi[1]:roi[3], roi[0]:roi[2]].flatten()
        n_sel = np.ones(z_sel.size)
        valid = z_sel > 0
        u_sel = u_sel[valid]
        v_sel = v_sel[valid]
        z_sel = z_sel[valid]
        n_sel = n_sel[valid]

        if u_sel.size > max_points:
            rng = np.random.default_rng()
            idx = rng.choice(u_sel.size, size=max_points, replace=False)
            u_sel = u_sel[idx]
            v_sel = v_sel[idx]
            z_sel = z_sel[idx]
            n_sel = n_sel[idx]

        z_sel = z_sel.astype(np.float32) / 1000.0  # Convert depth to meters
        uvn = np.vstack((u_sel, v_sel, n_sel))
        K_inv = np.linalg.inv(self.K)
        xyn = K_inv @ uvn
        xyn[0:2, :] = xyn[0:2, :] / xyn[2, :]
        xyz = xyn * z_sel
        return xyz.T

def main():
    rclpy.init()
    bridge = CvBridge()
    # Path to the ROS bag file
    bag_path = '/home/myrousz/rovozci_datasets/2024-03-07-Strancice-RGBD/rosbag2_2024_03_07-14_30_45/rosbag2_2024_03_07-14_30_45_0.db3'  # 必要に応じて更新
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    # Initialize the bag reader
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    # Retrieve topic types
    topic_type_pairs = reader.get_all_topics_and_types()
    topic_type_map = {topic.name: topic.type for topic in topic_type_pairs}
    topic_name = '/oak/stereo/image_raw/compressed'  
    if topic_name not in topic_type_map:
        print(f"[ERROR] Topic '{topic_name}' not found in the bag file.")
        return

    if topic_type_map[topic_name] != 'sensor_msgs/msg/CompressedImage':
        print(f"[ERROR] Topic '{topic_name}' is not of type CompressedImage.")
        return
    # Camera matrix (example) 2560x1440
    K_origin = np.array([
        [2067.0886, 0.0, 1235.7012],
        [0.0, 2065.7244, 727.0466],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    # Scaling factors
    scale_x = 1280 / 2560
    scale_y = 720 / 1440

    # Scaled camera parameters (720x1280)
    K = np.array([
        [K_origin[0, 0] * scale_x, 0.0, K_origin[0, 2] * scale_x],
        [0.0, K_origin[1, 1] * scale_y, K_origin[1, 2] * scale_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    generator = None  
    # Process messages from the bag
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == topic_name:
            try:
                # Deserialize the CompressedImage message
                msg = deserialize_message(data, CompressedImage)
                # Check if the data is valid
                if len(msg.data) == 0:
                    print(f"[WARN] Empty image data found in topic {topic}")
                    continue
                # Decode the compressed image using IMREAD_UNCHANGED
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                if cv_image is None:
                    print("[ERROR] Failed to decode the image.")
                    continue
                print("[INFO] Successfully decoded the image.")
                height, width = cv_image.shape[:2]
                if generator is None:
                    generator = PointCloudGenerator(K, width, height)
                # Example ROI and depth processing (update as needed)
                roi = (0, 0, width, height)
                points = generator.depth2points(cv_image, roi)
                print(f"[INFO] Generated {points.shape[0]} 3D points.")
                # Display the image (optional)
                cv2.imshow('Decoded Image', cv_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"[ERROR] Error processing topic {topic}: {e}")
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()