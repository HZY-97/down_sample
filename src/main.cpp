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
  // 检查输入点云是否为空
  if (!input_cloud || input_cloud->points.empty()) {
    LOG(ERROR) << "Input cloud is empty or null!";
    return;
  }

  // 设置输出点云的属性
  output_cloud->width = 0;
  output_cloud->height = 1;
  output_cloud->is_dense = true;

  int num_threads = thread_num;
  size_t num_points = input_cloud->points.size();

  // 1. 找到点云在X轴方向的边界
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();

  for (const auto& point : input_cloud->points) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
  }

  // 2. 计算每个线程负责的X轴范围
  float x_range = max_x - min_x;
  float x_step = x_range / num_threads;

  // 3. 根据X坐标将点云分配到不同的切片
  std::vector<PointCloudXYZI::Ptr> slice_clouds(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    slice_clouds[i].reset(new PointCloudXYZI());
  }

  for (const auto& point : input_cloud->points) {
    // 计算该点属于哪个线程
    int thread_id = static_cast<int>((point.x - min_x) / x_step);
    // 处理边界情况（max_x的点可能会超出范围）
    if (thread_id >= num_threads) {
      thread_id = num_threads - 1;
    }
    slice_clouds[thread_id]->points.push_back(point);
  }

  // 4. 创建线程池和结果存储
  std::vector<std::thread> threads;
  std::vector<PointCloudXYZI::Ptr> temp_results(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    temp_results[i].reset(new PointCloudXYZI());
  }

  // 5. 多线程处理
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      // 如果该切片为空，跳过
      if (slice_clouds[i]->points.empty()) {
        return;
      }

      // 创建八叉树
      pcl::octree::OctreePointCloudSearch<pcl::PointXYZINormal> octree(
          resolution);
      octree.setInputCloud(slice_clouds[i]);
      octree.addPointsFromInputCloud();

      // 获取所有叶节点的体素中心点
      std::vector<pcl::PointXYZINormal,
                  Eigen::aligned_allocator<pcl::PointXYZINormal>>
          voxel_centers;
      octree.getOccupiedVoxelCenters(voxel_centers);

      // 遍历每个体素
      for (const auto& centroid : voxel_centers) {
        std::vector<int> point_indices;
        if (octree.voxelSearch(centroid, point_indices)) {
          Eigen::Vector3f centroid_vector = centroid.getVector3fMap();

          // 找到距离质心最近的点
          float min_distance = std::numeric_limits<float>::max();
          int closest_point_index = -1;

          for (int index : point_indices) {
            Eigen::Vector3f point =
                slice_clouds[i]->points[index].getVector3fMap();
            float distance = (point - centroid_vector).squaredNorm();
            if (distance < min_distance) {
              min_distance = distance;
              closest_point_index = index;
            }
          }

          // 添加最近点到结果（不需要锁，因为每个线程写入自己的结果）
          if (closest_point_index != -1) {
            temp_results[i]->points.push_back(
                slice_clouds[i]->points[closest_point_index]);
          }
        }
      }
    });
  }

  // 6. 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 7. 合并所有结果
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