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

/**
 * @brief 使用八叉树进行点云下采样，选择距离质心最近的点作为每个叶节点的代表点。
 * @param input_cloud 输入点云指针
 * @param output_cloud 输出下采样后的点云指针
 * @param resolution 八叉树叶节点的分辨率（单位：米）
 */

void OctreeDownSample(PointCloudXYZI::Ptr& input_cloud,
                      PointCloudXYZI::Ptr& output_cloud, float resolution,
                      int thread_num = 1) {
  // 多线程操作
  int num_threads = thread_num;
  // 检查输入点云是否为空
  if (!input_cloud || input_cloud->points.empty()) {
    LOG(ERROR) << "Input cloud is empty or null!";
    return;
  }

  // 设置输出点云的属性
  output_cloud->width = 0;
  output_cloud->height = 1;
  output_cloud->is_dense = true;

  // 计算每个线程处理的点云大小
  size_t num_points = input_cloud->points.size();
  size_t points_per_thread = num_points / num_threads;

  // 创建一个线程池
  std::vector<std::thread> threads;

  // 创建一个临时点云以存储每个线程的结果
  std::vector<PointCloudXYZI::Ptr> temp_results(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    temp_results[i].reset(new PointCloudXYZI());
  }

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      // 每个线程创建自己的八叉树对象
      pcl::octree::OctreePointCloudSearch<pcl::PointXYZINormal> octree(
          resolution);

      // 每个线程处理自己的点云切片
      size_t start_index = i * points_per_thread;
      size_t end_index =
          (i == num_threads - 1) ? num_points : start_index + points_per_thread;

      // 获取当前线程的输入切片
      PointCloudXYZI::Ptr slice_cloud(new PointCloudXYZI);
      slice_cloud->points.insert(slice_cloud->points.end(),
                                 input_cloud->points.begin() + start_index,
                                 input_cloud->points.begin() + end_index);

      // 设置输入点云
      octree.setInputCloud(slice_cloud);
      octree.addPointsFromInputCloud();

      // 获取所有叶节点的体素中心点
      std::vector<pcl::PointXYZINormal,
                  Eigen::aligned_allocator<pcl::PointXYZINormal>>
          voxel_centers;
      octree.getOccupiedVoxelCenters(voxel_centers);

      // 遍历每个叶节点（体素中心点）
      for (const auto& centroid : voxel_centers) {
        std::vector<int> point_indices;

        // 查找体素中包含的所有点索引
        if (octree.voxelSearch(centroid, point_indices)) {
          // 将体素中心点转换为 Eigen::Vector3f 以便进行距离计算
          Eigen::Vector3f centroid_vector = centroid.getVector3fMap();

          // 找到距离质心最近的点
          float min_distance = std::numeric_limits<float>::max();
          int closest_point_index = -1;

          for (int index : point_indices) {
            Eigen::Vector3f point = slice_cloud->points[index].getVector3fMap();
            float distance = (point - centroid_vector).squaredNorm();
            if (distance < min_distance) {
              min_distance = distance;
              closest_point_index = index;
            }
          }

          // 将距离质心最近的点作为代表点添加到临时结果中
          if (closest_point_index != -1) {
            std::lock_guard<std::mutex> lock(octreeDownSample_mtx);
            temp_results[i]->points.push_back(
                slice_cloud->points[closest_point_index]);
          }
        }
      }
    });
  }

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 合并所有临时结果到输出点云中
  for (const auto& temp_result : temp_results) {
    output_cloud->points.insert(output_cloud->points.end(),
                                temp_result->points.begin(),
                                temp_result->points.end());
  }

  // 设置输出点云的属性
  output_cloud->width = output_cloud->points.size();
  output_cloud->height = 1;

  LOG(INFO) << "Octree sampling completed. Output point cloud size: "
            << output_cloud->points.size();
}

int main(int argc, char** argv) {
  // init glog
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_logbufsecs = 0;
  FLAGS_colorlogtostderr = true;
  FLAGS_logtostderr = true;

  if (argc < 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " <input_file>"
               << " 0.2";
    return -1;
  }
  std::string input_file = argv[1];
  double octreeDownSample_resolution = std::stod(argv[2]);
  // 定义点云文件的路径及文件名
  std::filesystem::path input_path = input_file;

  // pcl load pcd
  pcl::PCDReader reader;
  PointCloudXYZI::Ptr in_cloud(new PointCloudXYZI);

  reader.read(input_file, *in_cloud);
  LOG(INFO) << "Loaded " << in_cloud->size() << " points from " << input_file;

  PointCloudXYZI::Ptr out_cloud(new PointCloudXYZI);
  OctreeDownSample(in_cloud, out_cloud, octreeDownSample_resolution, 8);
  LOG(INFO) << "Downsampled " << out_cloud->size() << " points";
  pcl::PCDWriter writer;

  std::filesystem::path output_path =
      input_path.parent_path() / "downsampled_octomap.pcd";
  writer.write(output_path, *out_cloud);

  LOG(INFO) << "save " << output_path << " successfully";

  pcl::VoxelGrid<PointType> voxel_filter;
  voxel_filter.setInputCloud(in_cloud);
  voxel_filter.setLeafSize(
      octreeDownSample_resolution, octreeDownSample_resolution,
      octreeDownSample_resolution);  // 设置体素大小，与你的0.1参数对应

  PointCloudXYZI::Ptr downsampled_cloud(new PointCloudXYZI);
  voxel_filter.filter(*downsampled_cloud);

  LOG(INFO) << "pcl Downsampled point cloud size: "
            << downsampled_cloud->size();

  std::filesystem::path pcl_path =
      input_path.parent_path() / "downsampled_pcl.pcd";
  writer.write(pcl_path, *downsampled_cloud);
  LOG(INFO) << "save " << pcl_path << " successfully";

  return 0;
}