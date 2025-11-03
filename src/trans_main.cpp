/**
 * @file main.cpp
 * @brief 多线程ASC文件转换为PCD文件
 * @version 0.2
 * @date 2024-08-21
 *
 */

#include <glog/logging.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>

#include <Eigen/Core>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <pcl/impl/point_types.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

std::mutex octreeDownSample_mtx;

int main(int argc, char** argv) {
  // init glog
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_logbufsecs = 0;
  FLAGS_colorlogtostderr = true;
  FLAGS_logtostderr = true;

  if (argc < 2) {
    LOG(ERROR) << "Usage: " << argv[0] << " <input_file>";
    return -1;
  }
  std::string input_file = argv[1];
  // 定义点云文件的路径及文件名
  std::filesystem::path input_path = input_file;

  double x = 0, y = 0, z = 0, roll_deg = 0, pitch_deg = 0, yaw_deg = 0;
  std::cout << "input trans x: ";
  std::cin >> x;
  std::cout << "input trans y: ";
  std::cin >> y;
  std::cout << "input trans z: ";
  std::cin >> z;
  std::cout << "input trans roll_deg: ";
  std::cin >> roll_deg;
  std::cout << "input trans pitch_deg: ";
  std::cin >> pitch_deg;
  std::cout << "input trans yaw_deg:";
  std::cin >> yaw_deg;

  Eigen::Affine3f trans = Eigen::Affine3f::Identity();
  pcl::getTransformation(x, y, z, roll_deg * M_PI / 180.0,
                         pitch_deg * M_PI / 180.0, yaw_deg * M_PI / 180.0,
                         trans);

  // pcl load pcd
  pcl::PCDReader reader;
  PointCloudXYZI::Ptr in_cloud(new PointCloudXYZI);

  reader.read(input_file, *in_cloud);
  LOG(INFO) << "Loaded " << in_cloud->size() << " points from " << input_file;

  PointCloudXYZI::Ptr out_cloud(new PointCloudXYZI);

  pcl::transformPointCloud(*in_cloud, *out_cloud, trans);

  LOG(INFO) << "trans over: " << out_cloud->size() << " points";
  pcl::PCDWriter writer;

  std::filesystem::path output_path =
      input_path.parent_path() / "trans_over.pcd";
  writer.writeBinaryCompressed(output_path, *out_cloud);

  return 0;
}