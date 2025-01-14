import cv2
import numpy as np
from ultralytics import YOLO
import glob
import os
#画像保存あり
class BrickPoseEstimator:
    def __init__(self, model_path):
        # YOLOモデルの読み込み
        self.yolo = YOLO(model_path)

        # 元のカメラパラメータ (2K: 1440x2560)
        self.original_camera_matrix = np.array([
            [2067.0886, 0.0, 1235.7012],
            [0.0, 2065.7244, 727.0466],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([7.1634, -88.0136, -0.00047, -0.00045, 219.1458, 6.9479, -86.5780, 216.1373], dtype=np.float32)

        self._last_distances = []
        self._last_centers = []

    def scale_camera_matrix(self, image_height, image_width):
        # スケールファクターの計算
        height_scale = image_height / 1440
        width_scale = image_width / 2560

        # カメラ行列のスケーリング
        scaled_matrix = self.original_camera_matrix.copy()
        scaled_matrix[0, 0] *= width_scale   # fx
        scaled_matrix[1, 1] *= height_scale  # fy
        scaled_matrix[0, 2] *= width_scale   # cx
        scaled_matrix[1, 2] *= height_scale  # cy

        return scaled_matrix

    def process_image(self, image, depth_image_shape):
        # RGB画像を深度画像と同じ解像度にリサイズ
        image_resized = cv2.resize(image, (depth_image_shape[1], depth_image_shape[0]))

        # カメラ行列のスケーリング
        self.camera_matrix = self.scale_camera_matrix(depth_image_shape[0], depth_image_shape[1])

        # YOLO推論
        results = self.yolo(image_resized)[0]
        print(f"Image size: {image_resized.shape[1]}x{image_resized.shape[0]}")
        print(f"Scaled camera matrix:\n{self.camera_matrix}")

        self._last_distances = []
        self._last_centers = []

        if results.keypoints is not None:
            print(f"Detections found: {len(results.keypoints.data)}")

            for idx, (kpts, cls) in enumerate(zip(results.keypoints.data, results.boxes.cls)):
                class_id = int(cls)
                label = results.names[class_id]

                # 4つのキーポイントを持つBricksをフィルタリング
                if label == 'Bricks' and len(kpts) == 4:
                    # 信頼度のチェック
                    conf_scores = kpts[:, 2]
                    if all(conf > 0.5 for conf in conf_scores):
                        keypoints = kpts[:, :2].cpu().numpy()

                        try:
                            # 姿勢推定
                            distance, tvec, rvec, center, is_front = self.estimate_pose(keypoints)
                            self._last_distances.append(distance / 1000.0)  # メートル単位に変換
                            self._last_centers.append(center)

                            # 結果の描画（任意）
                            image_resized = self.draw_results(image_resized, keypoints, distance, tvec, rvec, is_front)
                        except Exception as e:
                            print(f"Error in pose estimation: {str(e)}")

        return image_resized

    def estimate_pose(self, keypoints_2d):
        def get_brick_parameters(points_2d, camera_matrix, dist_coeffs):
            # 正面と側面の寸法定義
            front = {'half_w': 500, 'half_h': 690}  # 正面: 幅100cm, 高さ138cm
            side = {'half_w': 610, 'half_h': 690}   # 側面: 幅122cm, 高さ138cm

            def get_3d_points(params):
                return np.float32([
                    [-params['half_w'], -params['half_h'], 0],
                    [ params['half_w'], -params['half_h'], 0],
                    [ params['half_w'],  params['half_h'], 0],
                    [-params['half_w'],  params['half_h'], 0]
                ])

            def compute_error(params):
                points_3d = get_3d_points(params)
                success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs)
                if not success:
                    return float('inf'), None, None

                projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
                error = np.mean(np.linalg.norm(points_2d - projected_points.reshape(-1, 2), axis=1))
                return error, rvec, tvec

            # 正面と側面の誤差を計算
            front_error, front_rvec, front_tvec = compute_error(front)
            side_error, side_rvec, side_tvec = compute_error(side)

            # 誤差が小さい方を選択
            if front_error < side_error:
                return front['half_w'], front['half_h'], front_rvec, front_tvec, True
            else:
                return side['half_w'], side['half_h'], side_rvec, side_tvec, False

        image_points = np.array(keypoints_2d, dtype=np.float32)
        half_w, half_h, rvec, tvec, is_front = get_brick_parameters(image_points, self.camera_matrix, self.dist_coeffs)
        distance = np.linalg.norm(tvec)

        # 中心点の計算
        center = np.mean(image_points, axis=0)

        return distance, tvec, rvec, center, is_front

    def draw_results(self, image, keypoints, distance, tvec, rvec, is_front):
        # キーポイントと線の描画
        color = (0, 255, 0) if is_front else (255, 0, 0)  # 正面: 緑, 側面: 青
        for pt in keypoints:
            x, y = map(int, pt)
            cv2.circle(image, (x, y), 5, color, -1)

        for i in range(len(keypoints)):
            pt1 = tuple(map(int, keypoints[i]))
            pt2 = tuple(map(int, keypoints[(i + 1) % len(keypoints)]))
            cv2.line(image, pt1, pt2, color, 2)

        # 中心点の計算
        center_x = int(np.mean(keypoints[:, 0]))
        center_y = int(np.mean(keypoints[:, 1]))

        # 中心点の描画
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

        # テキストの準備
        text = f"{distance / 1000.0:.3f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # テキストサイズの計算
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # テキストの位置
        text_x = center_x - text_width // 2
        text_y = center_y - 15

        # 背景の矩形描画
        padding = 4
        cv2.rectangle(image,
                      (text_x - padding, text_y - text_height - padding),
                      (text_x + text_width + padding, text_y + padding),
                      (0, 0, 0),
                      -1)

        # テキストの描画
        cv2.putText(image, text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness)

        return image

    def get_last_distances(self):
        """推定された距離のリストを返す（メートル単位）"""
        return self._last_distances

    def get_last_centers(self):
        """推定された中心点のリストを返す"""
        return self._last_centers

def calculate_actual_distance(depth_image, center, camera_matrix):
    """
    深度画像とカメラパラメータから実際の距離を計算する。

    Args:
        depth_image (np.ndarray): 深度画像
        center (np.ndarray): 推定された中心点の座標
        camera_matrix (np.ndarray): カメラ行列

    Returns:
        float: カメラからの距離（メートル単位）
    """
    cx, cy = int(center[0]), int(center[1])

    # インデックスが画像範囲内にあるかチェック
    if cx < 0 or cx >= depth_image.shape[1] or cy < 0 or cy >= depth_image.shape[0]:
        raise ValueError("Center point is out of image bounds")

    depth = depth_image[cy, cx] / 1000.0  # 深度値をメートルに変換

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cxo = camera_matrix[0, 2]
    cyo = camera_matrix[1, 2]

    x = (center[0] - cxo) * depth / fx
    y = (center[1] - cyo) * depth / fy
    z = depth

    return np.linalg.norm([x, y, z])

def save_result_image(rgb_path, result_image):
    # 元のサブフォルダの名前を取得
    subfolder_name = os.path.basename(os.path.dirname(rgb_path))
    # 結果画像の保存先ディレクトリを作成
    result_dir = os.path.join("results", subfolder_name)
    os.makedirs(result_dir, exist_ok=True)
    # 結果画像の保存パスを作成
    result_image_path = os.path.join(result_dir, os.path.basename(rgb_path).replace("rgb_frame_", "result_frame_"))
    # 結果画像を保存
    cv2.imwrite(result_image_path, result_image)
    print(f"Result image saved to: {result_image_path}")

def evaluate_distance_estimation(dataset_path):
    """
    データセットを使用して距離計算プログラムの精度評価を行う。

    Args:
        dataset_path (str): RGB画像、深度画像、およびポイントクラウドデータを含むフォルダのパス
    """
    # BrickPoseEstimatorの初期化
    model_path = "YOLO11/weight/2024-11-29-pose-yolo11-1000epoch(2024-11-27_vertical_aug).pt"
    estimator = BrickPoseEstimator(model_path)

    # RGB画像のファイルパスを取得
    rgb_files = glob.glob(os.path.join(dataset_path, "**/rgb_frame_*.png"), recursive=True)

    errors = []
    for rgb_path in rgb_files:
        # 対応する深度画像の読み込み
        depth_path = rgb_path.replace("rgb_frame_", "depth_frame_")
        if not os.path.exists(depth_path):
            continue
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        if depth_image is None:
            continue

        # RGB画像の読み込み
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue

        # 距離推定を実行
        result_image = estimator.process_image(rgb, depth_image.shape)
        estimated_distances = estimator.get_last_distances()
        centers = estimator.get_last_centers()

        if not estimated_distances:
            continue

        for estimated_distance, center in zip(estimated_distances, centers):
            try:
                # 深度画像から実際の距離を計算
                actual_distance = calculate_actual_distance(depth_image, center, estimator.camera_matrix)

                # 誤差を計算
                error = abs(estimated_distance - actual_distance)
                errors.append(error)

                print(f"Image: {os.path.basename(rgb_path)}")
                print(f"Estimated distance: {estimated_distance:.3f}m")
                print(f"Actual distance: {actual_distance:.3f}m")
                print(f"Error: {error:.3f}m")
                print("-" * 40)
            except ValueError as e:
                print(f"Error: {e}")

        # 結果画像を保存
        save_result_image(rgb_path, result_image)

    if errors:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        print(f"\nEvaluation Results:")
        print(f"Mean error: {mean_error:.3f}m")
        print(f"Std deviation: {std_error:.3f}m")

if __name__ == "__main__":
    evaluate_distance_estimation("C:/Users/syach/CIIRC_Pose-Estimation/dataset/pointscloud/test")