from setuptools import find_packages, setup

package_name = 'cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/model/keypoint_classifier', [
            'cv/model/keypoint_classifier/keypoint_classifier.tflite',
            'cv/model/keypoint_classifier/keypoint_classifier_label.csv',
        ]),
    ],
    install_requires=['setuptools',
                      'opencv-python',
                      'mediapipe',
                      'rclpy'],
    zip_safe=True,
    maintainer='thanhf',
    maintainer_email='thanhf@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_landmark_node = cv.test:main',
            'gesture_recognition_node = cv.cnn_test_3_ros:main',
            'chess_subscriber_node = cv.cnn_sub:main'
        ],
    },
    include_package_data=True,   # keep this True
)
