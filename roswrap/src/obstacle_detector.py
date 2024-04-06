import os
import plc
import rclpy
import yaml
import argparse

from pathlib import Path
from mmdet3d.apis import LidarDet3DInferencer
from rclpy.node import Node
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2, PointField
from roswrap.msg import mmdet3D_inference, mmdet3D_object


class PointCloudSubscriber(Node):
    def __init__(self, cfg):
        super().__init__('point_cloud_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            cfg['obstacle_detector']['pcd_topic'],
            self.pointcloud_callback,
            1)
        
        self.pcl_data = None

    def pointcloud_callback(self, msg):
        plc_data = plc.PointCloud(msg)
        self.pcl_data = plc_data
         

class DetectionPublisher(Node):
    def __init__(self, cfg):
        super().__init__('detection_publisher')
        self.publisher = self.create_publisher(
            mmdet3D_inference,
            cfg['obstacle_detector']['detection_topic'],
            10)
        
        self.visualization_publisher = self.create_publisher(
            PointCloud2,
            cfg['obstacle_detector']['visualization_topic'],
            10)
        
    def publish_detection(self, _inference_msg):
        self.publisher.publish(_inference_msg)

    def publish_visualization(self, _visualization_msg):
        self.visualization_publisher.publish(_visualization_msg)
    

class ObstacleDetector(Node):
    def __init__(self, _args):
        super().__init__('obstacle_detector')
        _path = self.resolve_path(_args.cfg)
        self.load_config(_path)

        self.pointcloud_subscriber = PointCloudSubscriber(cfg=self.config)
        self.detection_publisher = DetectionPublisher(cfg=self.config)

        self.inferencer = LidarDet3DInferencer(**self.config['model_name'])

        # Wait for first point cloud message
        while rclpy.ok() and self.pointcloud_subscriber.pcl_data is None:
            rclpy.spin_once(self.pointcloud_subscriber)
            print('[ObstacleDetector] Waiting for point cloud message...')

        # Publisher timer
        frequency = self.config['obstacle_detector']['publish_frequency']
        self.timer = self.create_timer(1/frequency, self.obstacle_detector_callback)

    def load_config(self, _path):
        with open(_path) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def resolve_path(self, _path):
        path = Path(_path)
        if os.path.exists(path):
            return path
        elif path.resolve().exists():
            return path.resolve()
        else:
            raise FileNotFoundError(f'File not found: {path}')
            
    def detect_obstacles(self):
        point_cloud = self.pointcloud_subscriber.pcl_data
        detections = self.inferencer(point_cloud, show=False)
        return point_cloud, detections
    
    def generate_inference_msg(self, _pcd_msg, _detections):

        # TODO: For a flawless implenetation we should create a config file that can be used to specify the fields
        #       of the mmdet3D_prediction message for the according model.

        inference_msg = mmdet3D_inference()
        inference_msg.header = Header()
        inference_msg.header.stamp = _pcd_msg.header.stamp
        inference_msg.header.frame_id = 'LiDAR'
        prediction_array = []
        num_predictions = len(_detections[0]['labels_3d'])
        for i in range(num_predictions):
            object = mmdet3D_object()
            object.label_id = _detections[0]['labels_3d'][i]
            object.label = self.labels[_detections[0]['labels_3d'][i]]
            object.score = _detections[0]['scores_3d'][i]
            object.bbox = _detections[0]['boxes_3d'][i]
            prediction_array.append(object)
        inference_msg.predictions = prediction_array
        return inference_msg

    def publish_detections(self, _inference_msg):
        self.detection_publisher.publish_detection(_inference_msg)

    def obstacle_detector_callback(self):
        pcd_msg, detections = self.detect_obstacles()
        inference_msg = self.generate_inference_msg(pcd_msg, detections)
        self.publish_detections(inference_msg)

        if self.config['obstacle_detector']['publish_visualization']:
            visualization_msg = self.generate_visualization_msg(detections)
            self.detection_publisher.publish_visualization(visualization_msg)

def main(args=None):
    rclpy.init(args=args)

    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', 
                        type=str, 
                        default='../config/obstacle_detector.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    obstacle_detector = ObstacleDetector(args)
    rclpy.spin(obstacle_detector)
    obstacle_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()